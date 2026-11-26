"""ACL Mechanistic Evidence Suite on Qwen3.5-9B (GPU 5):
  M1: Real empirical Gated DeltaNet cumulative retention prod(alpha_t) under v1 vs sandwich_v3
  M2: End-to-end A800 latency, throughput, token length & VRAM benchmark (v1 vs sandwich_v3)
  M3: Generative Thinking/CoT baseline accuracy & latency on stratified 60-example holdout
"""
import json, math, re, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet

from nimble.scoring.parallel_schema import prepare_prompts
from nimble.training.schema_data import read_rows, as_scoring

root = Path("/data/nvme0/nimble_p9_9b")
MODEL = "Qwen/Qwen3.5-9B"
REV = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
out_dir = root / "acl_analysis"
out_dir.mkdir(exist_ok=True)

print("[1/3] Loading Qwen3.5-9B on CUDA...", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL, revision=REV, local_files_only=True)
base = Qwen3_5ForConditionalGeneration.from_pretrained(
    MODEL, revision=REV, dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True
)
if hasattr(base, "model") and hasattr(base.model, "visual"):
    del base.model.visual
base = base.to("cuda").eval()
base.config.use_cache = False
backbone = base.model

# Load 64 diverse long-context rows from eval.jsonl (sorted by length, top 64 + spread)
raw_eval = read_rows(root / "data/eval.jsonl")

def make_prompt_ids(row, fmt):
    sc = as_scoring(row, training=False)
    prep = prepare_prompts(tok, sc["context"], sc["schema"], 2048, system_role=True, prompt_format=fmt)
    return prep.full_ids[0], prep.candidate_ids[0], prep.choices[0], prep.names[0], sc

# --- M1: Empirical Gated DeltaNet retention prod_{t=t0+1..T} alpha_t ---
print("[M1] Measuring real Gated DeltaNet cumulative retention across all 24 DeltaNet layers...", flush=True)
deltanet_layers = [(name, mod) for name, mod in backbone.named_modules() if isinstance(mod, Qwen3_5GatedDeltaNet)]
print(f"Found {len(deltanet_layers)} Qwen3_5GatedDeltaNet layers", flush=True)

captured_g = {}
hooks = []
def make_hook(layer_name, parent_mod):
    def fn(submod, inp, out):
        with torch.no_grad():
            a = out.float()  # (B, T, n_heads) directly from in_proj_a
            g = -parent_mod.A_log.float().exp() * F.softplus(a + parent_mod.dt_bias.float())
            captured_g[layer_name] = g[0].detach().cpu()
    return fn

for lname, mod in deltanet_layers:
    hooks.append(mod.in_proj_a.register_forward_hook(make_hook(lname, mod)))

def find_choices_midpoint(ids, fmt):
    text = tok.decode(ids)
    # Locate start of Choices / Schema choices in text and map approximately to token index
    if fmt == "v1":
        # In v1, choices live inside JSON schema near the start, before "context"
        ctx_pos = text.find('"context"')
        if ctx_pos < 0:
            ctx_pos = text.find('## Context')
        ch_pos = text.find('"choices"')
        if ch_pos < 0:
            ch_pos = max(0, ctx_pos // 2) if ctx_pos > 0 else len(text) // 6
        mid_char = (ch_pos + max(ch_pos + 1, ctx_pos)) // 2 if ctx_pos > ch_pos else ch_pos
    else:
        # In sandwich_v3, choices live after "## Choices" near the tail
        ch_pos = text.find("## Choices")
        req_pos = text.find("Requested field:")
        if ch_pos < 0:
            ch_pos = int(len(text) * 0.85)
        if req_pos < ch_pos:
            req_pos = len(text) - 10
        mid_char = (ch_pos + req_pos) // 2
    # Exact token index via prefix encoding
    pref_len = len(tok.encode(text[:mid_char], add_special_tokens=False))
    return min(max(pref_len, 1), len(ids) - 2)

# Sample 60 rows stratified across lengths (20 short, 20 medium, 20 long)
lengths_v1 = [(len(make_prompt_ids(r, "v1")[0]), idx) for idx, r in enumerate(raw_eval)]
lengths_v1.sort()
n_tot = len(lengths_v1)
sample_idxs = (
    [lengths_v1[i][1] for i in np.linspace(0, n_tot // 3 - 1, 20, dtype=int)] +
    [lengths_v1[i][1] for i in np.linspace(n_tot // 3, 2 * n_tot // 3 - 1, 20, dtype=int)] +
    [lengths_v1[i][1] for i in np.linspace(2 * n_tot // 3, n_tot - 1, 20, dtype=int)]
)

m1_results = {"v1": [], "sandwich_v3": []}
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    for fmt in ("v1", "sandwich_v3"):
        for idx in sample_idxs:
            ids, _, _, _, _ = make_prompt_ids(raw_eval[idx], fmt)
            t_mid = find_choices_midpoint(ids, fmt)
            dist = len(ids) - 1 - t_mid
            captured_g.clear()
            inp_t = torch.tensor([ids], device="cuda")
            _ = backbone(input_ids=inp_t, use_cache=False)
            layer_ret = []
            layer_log_ret = []
            for lname, _ in deltanet_layers:
                g = captured_g[lname]  # (T, H)
                cum_log = g[t_mid + 1 :].sum(dim=0)  # (H,)
                ret = torch.exp(cum_log.clamp(min=-80.0))  # (H,)
                layer_ret.append(float(ret.mean().item()))
                layer_log_ret.append(float(cum_log.mean().item()))
            m1_results[fmt].append({
                "idx": int(idx),
                "seq_len": len(ids),
                "t_mid": int(t_mid),
                "distance": int(dist),
                "mean_retention": float(np.mean(layer_ret)),
                "median_layer_retention": float(np.median(layer_ret)),
                "mean_log_retention": float(np.mean(layer_log_ret)),
                "per_layer_retention": layer_ret,
            })

for h in hooks:
    h.remove()

m1_summary = {}
for fmt in ("v1", "sandwich_v3"):
    rows = m1_results[fmt]
    arr_ret = np.array([r["mean_retention"] for r in rows])
    arr_dist = np.array([r["distance"] for r in rows])
    arr_len = np.array([r["seq_len"] for r in rows])
    per_layer_mean = np.mean([r["per_layer_retention"] for r in rows], axis=0).tolist()
    # By length tertile
    buckets = {}
    for label, lo, hi in [("<512", 0, 512), ("512-1024", 512, 1024), (">=1024", 1024, 99999)]:
        sub = [r for r in rows if lo <= r["seq_len"] < hi]
        if sub:
            buckets[label] = {
                "n": len(sub),
                "mean_distance": float(np.mean([r["distance"] for r in sub])),
                "mean_retention": float(np.mean([r["mean_retention"] for r in sub])),
                "median_retention": float(np.median([r["mean_retention"] for r in sub])),
            }
    m1_summary[fmt] = {
        "n": len(rows),
        "mean_seq_len": float(arr_len.mean()),
        "mean_distance": float(arr_dist.mean()),
        "mean_retention": float(arr_ret.mean()),
        "median_retention": float(np.median(arr_ret)),
        "by_length_bucket": buckets,
        "per_layer_mean_retention": per_layer_mean,
    }
print("M1 Summary:", json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "per_layer_mean_retention"} for k, v in m1_summary.items()}, indent=2), flush=True)

# --- M2: End-to-End Latency, Token Length & Peak VRAM (v1 vs sandwich_v3) ---
print("\n[M2] Benchmarking A800 end-to-end latency & token savings (324 Holdout)...", flush=True)
m2_summary = {}
head_w = base.get_output_embeddings().weight
for fmt in ("v1", "sandwich_v3"):
    encoded = [make_prompt_ids(r, fmt) for r in raw_eval]
    tok_lens = [len(e[0]) for e in encoded]
    # Warmup 5
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for e in encoded[:5]:
            t = torch.tensor([e[0]], device="cuda")
            _ = backbone(input_ids=t, use_cache=False).last_hidden_state[:, -1, :]
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    latencies = []
    t_all0 = time.perf_counter()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for e in encoded:
            t0 = time.perf_counter()
            inp_t = torch.tensor([e[0]], device="cuda")
            h = backbone(input_ids=inp_t, use_cache=False).last_hidden_state[:, -1, :]
            cand = torch.tensor(e[1], device="cuda")
            _ = (h.float() @ head_w[cand].float().T)
            torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000.0)
    total_s = time.perf_counter() - t_all0
    m2_summary[fmt] = {
        "n": len(encoded),
        "mean_tokens": float(np.mean(tok_lens)),
        "p95_tokens": float(np.percentile(tok_lens, 95)),
        "mean_ms": float(np.mean(latencies)),
        "p50_ms": float(np.percentile(latencies, 50)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "throughput_req_s": len(encoded) / total_s,
        "peak_vram_gb": torch.cuda.max_memory_allocated() / (1024**3),
    }
tok_saving = (1.0 - m2_summary["sandwich_v3"]["mean_tokens"] / m2_summary["v1"]["mean_tokens"]) * 100.0
lat_speedup = m2_summary["v1"]["mean_ms"] / m2_summary["sandwich_v3"]["mean_ms"]
m2_summary["token_reduction_pct"] = tok_saving
m2_summary["latency_speedup_x"] = lat_speedup
print("M2 Summary:", json.dumps(m2_summary, indent=2), flush=True)

# --- M3: Generative Zero-Shot CoT / Thinking Baseline on 60 Stratified Holdout Examples ---
print("\n[M3] Benchmarking Qwen3.5-9B Generative CoT (Thinking Mode, max_new_tokens=256) on 60 stratified examples...", flush=True)
base.config.use_cache = True
by_kind = {"choice": [], "noul": [], "score": []}
for idx, r in enumerate(raw_eval):
    sc0 = as_scoring(r, training=False)
    k = sc0.get("kind") or r.get("kind") or ("noul" if next(iter(sc0["schema"].values()))["type"] == "boolean" else ("score" if all(str(c).isdigit() for c in next(iter(sc0["schema"].values()))["choices"]) else "choice"))
    by_kind[k].append(idx)
cot_idxs = by_kind["choice"][:20] + by_kind["noul"][:20] + by_kind["score"][:20]

cot_rows = []
cot_lats = []
cot_gen_toks = []
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    for i, idx in enumerate(cot_idxs):
        r = raw_eval[idx]
        sc = as_scoring(r, training=False)
        field_name = next(iter(sc["schema"]))
        field_def = sc["schema"][field_name]
        choices = field_def.get("choices", [False, True])
        codes = [chr(ord("A") + j) for j in range(len(choices))]
        descs = field_def.get("choice_descriptions", {})
        ch_lines = []
        for code, val in zip(codes, choices):
            vk = str(val).lower() if isinstance(val, bool) else str(val)
            d = descs.get(vk, "")
            ch_lines.append(f"- {code} ({vk}): {d}" if d else f"- {code}: {vk}")
        user_msg = (
            f"Field: {field_name}\nInstructions: {field_def['description']}\n\n"
            f"Context:\n{sc['context']}\n\nChoices:\n" + "\n".join(ch_lines) +
            "\n\nReason concisely in 2-3 short sentences based only on facts in the context, then end on a new line with exactly: Final Answer: <ONE_LETTER_CODE>"
        )
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": user_msg}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        inp = tok(prompt, return_tensors="pt", truncation=True, max_length=2048).to("cuda")
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = base.generate(**inp, max_new_tokens=256, do_sample=False, pad_token_id=tok.eos_token_id)
        torch.cuda.synchronize()
        dt_ms = (time.perf_counter() - t0) * 1000.0
        gen_ids = out[0][inp["input_ids"].shape[1]:]
        gen_txt = tok.decode(gen_ids, skip_special_tokens=True)
        # Parse final letter
        m = re.findall(r"Final Answer:\s*([A-Z])\b", gen_txt)
        if not m:
            # Fallback: last standalone capital letter matching valid codes after </think>
            tail = gen_txt.split("</think>")[-1] if "</think>" in gen_txt else gen_txt[-80:]
            m = re.findall(r"\b([" + "".join(codes) + r"])\b", tail)
        pred_code = m[-1] if m else None
        gold_val = sc.get("target", r.get("target"))
        gold_idx = sc["label_index"] if "label_index" in sc else choices.index(gold_val)
        gold_code = codes[gold_idx]
        ok = (pred_code == gold_code)
        cot_lats.append(dt_ms)
        cot_gen_toks.append(len(gen_ids))
        cot_rows.append({"idx": int(idx), "gold": gold_code, "pred": pred_code, "correct": bool(ok), "ms": dt_ms, "gen_tokens": len(gen_ids)})
        if (i + 1) % 10 == 0:
            print(f"  CoT {i+1}/60 | running acc={sum(x['correct'] for x in cot_rows)/(i+1)*100:.1f}% | mean_ms={np.mean(cot_lats):.0f}ms", flush=True)

m3_summary = {
    "n": len(cot_rows),
    "accuracy": sum(x["correct"] for x in cot_rows) / len(cot_rows),
    "parsed_rate": sum(x["pred"] is not None for x in cot_rows) / len(cot_rows),
    "mean_ms": float(np.mean(cot_lats)),
    "p50_ms": float(np.percentile(cot_lats, 50)),
    "mean_gen_tokens": float(np.mean(cot_gen_toks)),
    "speedup_single_token_v3_vs_cot": float(np.mean(cot_lats)) / m2_summary["sandwich_v3"]["mean_ms"],
}
print("M3 CoT Summary:", json.dumps(m3_summary, indent=2), flush=True)

full_out = {"M1_deltanet_retention": m1_summary, "M2_latency_efficiency": m2_summary, "M3_cot_baseline": m3_summary, "M3_rows": cot_rows}
(out_dir / "acl_mechanistic_report.json").write_text(json.dumps(full_out, indent=2))
print("MECHANISTIC_SUITE_DONE=True", flush=True)
