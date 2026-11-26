"""Evaluate a Nimble PEFT/base checkpoint on all 13 human-labeled public benchmarks (3,880 rows)."""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, Qwen3_5ForConditionalGeneration

from nimble.evaluation import evaluate_public
from nimble.scoring.cuda_scorer import candidate_projection, execution_device, physical_weight
from nimble.scoring.parallel_schema import choice_key, prepare_prompts
from nimble.scoring.serving_schema import codes_for

SUBSETS = [
    "aegis2",
    "boolq",
    "civil_comments",
    "helpsteer2",
    "massive-de-DE",
    "massive-en-US",
    "multinli",
    "paws",
    "pubmedqa",
    "squad2",
    "summeval-consistency",
    "summeval-relevance",
    "vitaminc-dev",
]


class PeftPublicScorer:
    def __init__(self, model_id, revision, adapter_dir=None, prompt_format="v1",
                 max_input_tokens=4096, drop_vision=False, bf16_full_lm_head=False):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU required")
        torch.backends.cuda.matmul.allow_tf32 = False
        self.model_id = model_id
        self.revision = revision
        self.prompt_format = prompt_format
        self.max_input_tokens = max_input_tokens
        self.bf16_full_lm_head = bf16_full_lm_head
        self.temperature = 1.0

        t0 = time.perf_counter()
        cfg = AutoConfig.from_pretrained(model_id, revision=revision, local_files_only=True)
        cls = Qwen3_5ForConditionalGeneration if getattr(cfg, "model_type", "") == "qwen3_5" else AutoModelForCausalLM
        base = cls.from_pretrained(
            model_id,
            revision=revision,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            local_files_only=True,
        )
        if drop_vision and hasattr(base, "model") and hasattr(base.model, "visual"):
            del base.model.visual
        base = base.to("cuda")
        if adapter_dir:
            peft_model = PeftModel.from_pretrained(base, str(adapter_dir))
            model = peft_model.merge_and_unload()
        else:
            model = base
        model.config.use_cache = False
        self.model = model.eval()
        self.backbone = self.model.model
        self.head_weight = physical_weight(self.model.get_output_embeddings())
        self.device = execution_device(self.model.get_input_embeddings())
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, revision=revision, local_files_only=True
        )
        self.runtime = {
            "dtype": "bfloat16",
            "attention": "sdpa",
            "prompt_format": prompt_format,
            "adapter_dir": str(adapter_dir) if adapter_dir else None,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "load_seconds": round(time.perf_counter() - t0, 2),
        }
        print(f"Loaded {model_id} (adapter={adapter_dir}, fmt={prompt_format}) in {self.runtime['load_seconds']}s on {self.runtime['gpu']}", flush=True)

    @torch.inference_mode()
    def score(self, context, schema, mode="independent"):
        if mode != "independent":
            raise ValueError("Only independent mode supported")
        prepared = prepare_prompts(
            self.tokenizer, context, schema, self.max_input_tokens,
            system_role=True, prompt_format=self.prompt_format,
        )
        started = time.perf_counter()
        fields, output = {}, {}
        for name, choices, ids, candidates in zip(
            prepared.names, prepared.choices, prepared.full_ids, prepared.candidate_ids
        ):
            tokens = torch.tensor([ids], device=self.device)
            if self.bf16_full_lm_head:
                logits_vocab = self.model(input_ids=tokens, use_cache=False, logits_to_keep=1).logits[:, -1, :].float()
                cand_idx = torch.tensor(candidates, device=logits_vocab.device, dtype=torch.long)
                logits = logits_vocab[0].index_select(0, cand_idx)
                hidden = logits_vocab
            else:
                hidden = self.backbone(input_ids=tokens, use_cache=False).last_hidden_state[:, -1, :]
                logits = candidate_projection(hidden, self.head_weight, candidates)[0]
            if not torch.isfinite(logits).all():
                raise ValueError("Non-finite candidate logits")
            probabilities = torch.softmax(logits, dim=-1)
            best = logits.argmax().item()
            keys = [choice_key(value) for value in choices]
            output[name] = choices[best]
            fields[name] = {
                "value": choices[best],
                "scores": dict(zip(keys, probabilities.tolist())),
                "logits": dict(zip(keys, logits.tolist())),
                "candidate_token_ids": candidates,
                "code_to_choice": dict(zip(codes_for(len(choices), self.tokenizer), choices)),
                "prompt_token_count": len(ids),
                "prompt_token_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
            }
            del hidden, logits, probabilities
        return {
            "model": self.model_id,
            "revision": self.revision,
            "backend": "cuda",
            "temperature": 1.0,
            "temperature_fitted": False,
            "runtime": self.runtime,
            "context": context,
            "output": output,
            "fields": fields,
            "metrics": {
                "mode": mode,
                "fields": len(fields),
                "total_seconds": time.perf_counter() - started,
                "full_vocabulary_projection": self.bf16_full_lm_head,
            },
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--prompt-format", default="v1", choices=["v1", "sandwich_v2", "sandwich_v3", "sandwich_v3_rawjson", "context_first_clean", "schema_first_clean"])
    ap.add_argument("--drop-vision", action="store_true")
    ap.add_argument("--bf16-full-lm-head", action="store_true")
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--data-root", type=Path, default=Path("/root/projects/nimble/data/public"))
    ap.add_argument("--out-root", type=Path, default=Path("/root/projects/nimble_p7_4b/eval_public"))
    ap.add_argument("--max-input-tokens", type=int, default=4096)
    args = ap.parse_args()

    scorer = PeftPublicScorer(
        model_id=args.model,
        revision=args.revision,
        adapter_dir=args.adapter,
        prompt_format=args.prompt_format,
        max_input_tokens=args.max_input_tokens,
        drop_vision=args.drop_vision,
        bf16_full_lm_head=args.bf16_full_lm_head,
    )

    suite_t0 = time.perf_counter()
    for sub in SUBSETS:
        data_file = args.data_root / sub / "all.jsonl"
        out_dir = args.out_root / sub / args.run_name
        if (out_dir / "summary.json").exists():
            s = json.loads((out_dir / "summary.json").read_text())
            acc = s["summary"]["all"]["accuracy"] * 100
            print(f"[SKIP] {args.run_name} / {sub}: already done (Acc={acc:.2f}%)", flush=True)
            continue
        t0 = time.perf_counter()
        res = evaluate_public.run(
            data=data_file,
            output=out_dir,
            model_path=args.adapter,
            model_id=args.model,
            revision=args.revision,
            backend="cuda",
            temperature=1.0,
            max_input_tokens=args.max_input_tokens,
            scorer_factory=lambda **kw: scorer,
        )
        acc = res["summary"]["all"]["accuracy"] * 100
        brier = res["summary"]["all"]["mean_multiclass_brier"]
        nll = res["summary"]["all"]["mean_negative_log_likelihood"]
        ece = res["calibration"]["ece10"]
        dt = time.perf_counter() - t0
        print(
            f"[DONE] {args.run_name} / {sub:<22} n={res['count']:3d} | "
            f"Acc={acc:5.2f}% NLL={nll:.4f} Brier={brier:.4f} ECE={ece:.4f} ({dt:.1f}s)",
            flush=True,
        )
    print(f"SUITE_COMPLETE {args.run_name} total_seconds={time.perf_counter() - suite_t0:.1f}", flush=True)


if __name__ == "__main__":
    main()
