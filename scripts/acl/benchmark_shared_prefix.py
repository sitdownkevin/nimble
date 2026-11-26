"""Benchmark CUDA Shared-Prefix (Hybrid DeltaNet + KV Cache Forking) vs Independent Multi-Field Scoring.

Tests F in {1, 2, 4, 8} concurrent schema fields per document on Qwen3.5-9B (sandwich_v3 vs v1),
measuring:
  - Prefix fraction (% of full prompt tokens shared across all F fields)
  - End-to-end latency (ms/doc) & speedup (Independent / Shared-Prefix)
  - Numerical equivalence: max |Delta prob| and argmax agreement (%) between Shared-Prefix and Independent!
"""

import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

from nimble.scoring.parallel_schema import prepare_prompts
from nimble.training.schema_data import as_scoring


def clone_hybrid_cache(past_kv, batch_size: int):
    """Deep-copy / batch-expand Qwen3.5 Hybrid DynamicCache (LinearAttentionLayer dicts + DynamicLayer KV)."""
    cloned = copy.deepcopy(past_kv)
    if batch_size == 1:
        return cloned
    for lyr in getattr(cloned, "layers", []):
        if hasattr(lyr, "batch_repeat_interleave"):
            lyr.batch_repeat_interleave(batch_size)
        else:
            for attr in ("conv_states", "recurrent_states"):
                d = getattr(lyr, attr, None)
                if isinstance(d, dict):
                    for k, t in list(d.items()):
                        if isinstance(t, torch.Tensor):
                            d[k] = t.repeat_interleave(batch_size, dim=0).contiguous()
    return cloned


def build_multifield_schema(base_schema: dict, n_fields: int) -> dict:
    """Replicate a realistic field schema into F distinct named decision fields on the same context."""
    first_name = next(iter(base_schema))
    base_def = base_schema[first_name]
    if n_fields == 1:
        return {first_name: base_def}
    out = {}
    for k in range(n_fields):
        out[f"{first_name}_aspect_{k+1}"] = copy.deepcopy(base_def)
    return out


@torch.inference_mode()
def main():
    model_id = "Qwen/Qwen3.5-9B"
    rev = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
    torch.backends.cuda.matmul.allow_tf32 = False

    tok = AutoTokenizer.from_pretrained(model_id, revision=rev, local_files_only=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_id, revision=rev, dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True
    )
    if hasattr(model, "model") and hasattr(model.model, "visual"):
        del model.model.visual
    model = model.to("cuda").eval()
    backbone = model.model.language_model
    head_w = model.get_output_embeddings().weight

    raw_rows = [json.loads(line) for line in Path("data/eval.jsonl").read_text().splitlines() if line.strip()]
    # Select 40 representative long/medium documents stratified across eval.jsonl
    step = max(1, len(raw_rows) // 40)
    sample_rows = raw_rows[::step][:40]

    # Warmup GPU
    w_ids = torch.tensor([[1, 2, 3, 4, 5]], device="cuda")
    for _ in range(3):
        _ = backbone(input_ids=w_ids, use_cache=True)
    torch.cuda.synchronize()

    results = {}
    for fmt in ("v1", "sandwich_v3"):
        results[fmt] = {}
        for n_fields in (1, 2, 4, 8):
            ind_ms, sp_serial_ms, sp_batch_ms = [], [], []
            prefix_ratios, max_dp_serial, max_dp_batch = [], [], []
            agree_serial, agree_batch, total_fields = 0, 0, 0

            for r in sample_rows:
                res_row = as_scoring(r, training=False)
                ctx, single_schema = res_row["context"], res_row["schema"]
                if isinstance(single_schema, str):
                    single_schema = json.loads(single_schema)
                schema = build_multifield_schema(single_schema, n_fields)
                prep = prepare_prompts(tok, ctx, schema, max_input_tokens=4096, prompt_format=fmt)
                mean_full = float(np.mean([len(x) for x in prep.full_ids]))
                prefix_ratios.append(len(prep.prefix_ids) / mean_full)

                # 1. Independent full-prompt prefill (baseline)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                ind_probs = []
                for ids, cands in zip(prep.full_ids, prep.candidate_ids):
                    inp = torch.tensor([ids], device="cuda")
                    h = backbone(input_ids=inp, use_cache=False).last_hidden_state[:, -1, :]
                    logits = (h.float() @ head_w[cands].float().T)[0]
                    ind_probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
                torch.cuda.synchronize()
                ind_ms.append((time.perf_counter() - t0) * 1000.0)

                # 2. Shared-Prefix Cached Serial (1 prefill + F short suffix steps)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                p_inp = torch.tensor([prep.prefix_ids], device="cuda")
                p_out = backbone(input_ids=p_inp, use_cache=True)
                prefix_kv = p_out.past_key_values
                ser_probs = []
                for suf, cands in zip(prep.suffix_ids, prep.candidate_ids):
                    kv_copy = clone_hybrid_cache(prefix_kv, batch_size=1)
                    s_inp = torch.tensor([suf], device="cuda")
                    h = backbone(input_ids=s_inp, past_key_values=kv_copy, use_cache=True).last_hidden_state[:, -1, :]
                    logits = (h.float() @ head_w[cands].float().T)[0]
                    ser_probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
                torch.cuda.synchronize()
                sp_serial_ms.append((time.perf_counter() - t0) * 1000.0)

                # 3. Shared-Prefix Batched Suffixes (1 prefill + 1 batched suffix forward!)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                p_inp = torch.tensor([prep.prefix_ids], device="cuda")
                p_out = backbone(input_ids=p_inp, use_cache=True)
                prefix_kv = p_out.past_key_values
                pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
                lengths = [len(s) for s in prep.suffix_ids]
                max_w = max(lengths)
                padded = torch.tensor([s + [pad_id] * (max_w - len(s)) for s in prep.suffix_ids], device="cuda")
                kv_batch = clone_hybrid_cache(prefix_kv, batch_size=n_fields)
                h_all = backbone(input_ids=padded, past_key_values=kv_batch, use_cache=True).last_hidden_state
                bat_probs = []
                for idx_f, (slen, cands) in enumerate(zip(lengths, prep.candidate_ids)):
                    h_f = h_all[idx_f : idx_f + 1, slen - 1, :]
                    logits = (h_f.float() @ head_w[cands].float().T)[0]
                    bat_probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
                torch.cuda.synchronize()
                sp_batch_ms.append((time.perf_counter() - t0) * 1000.0)

                for p_i, p_s, p_b in zip(ind_probs, ser_probs, bat_probs):
                    max_dp_serial.append(float(np.max(np.abs(p_i - p_s))))
                    max_dp_batch.append(float(np.max(np.abs(p_i - p_b))))
                    agree_serial += int(np.argmax(p_i) == np.argmax(p_s))
                    agree_batch += int(np.argmax(p_i) == np.argmax(p_b))
                    total_fields += 1

            m_ind = float(np.mean(ind_ms))
            m_ser = float(np.mean(sp_serial_ms))
            m_bat = float(np.mean(sp_batch_ms))
            entry = {
                "fields": n_fields,
                "prefix_share_pct": round(float(np.mean(prefix_ratios)) * 100, 2),
                "independent_ms": round(m_ind, 2),
                "shared_serial_ms": round(m_ser, 2),
                "shared_batched_ms": round(m_bat, 2),
                "speedup_serial": round(m_ind / m_ser, 2),
                "speedup_batched": round(m_ind / m_bat, 2),
                "max_abs_dp_serial": round(float(np.max(max_dp_serial)), 6),
                "max_abs_dp_batched": round(float(np.max(max_dp_batch)), 6),
                "argmax_agree_serial_pct": round(agree_serial / total_fields * 100, 2),
                "argmax_agree_batch_pct": round(agree_batch / total_fields * 100, 2),
            }
            results[fmt][str(n_fields)] = entry
            print(
                f"[{fmt:11s} | F={n_fields}] prefix={entry['prefix_share_pct']:5.1f}% | "
                f"Ind={m_ind:6.1f}ms | SharedSerial={m_ser:6.1f}ms ({entry['speedup_serial']:.2f}x) | "
                f"SharedBatch={m_bat:6.1f}ms ({entry['speedup_batched']:.2f}x) | "
                f"max|dp|={entry['max_abs_dp_batched']:.5f} agree={entry['argmax_agree_batch_pct']:.1f}%",
                flush=True,
            )

    out_p = Path("acl_analysis/shared_prefix_report.json")
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(results, indent=2) + "\n")
    print(f"SHARED_PREFIX_BENCH_DONE=True -> {out_p}", flush=True)


if __name__ == "__main__":
    main()
