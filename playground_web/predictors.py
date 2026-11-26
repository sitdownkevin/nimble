"""Multi-method predictor pool for experiencing Nimble SOTA vs Legacy vs Kev in the Web Playground.

Shares base Qwen3.5 backbones across multiple PEFT adapters via ``load_adapter`` / ``set_adapter`` so 8+ models
coexist comfortably on a single GPU with <0.5ms switching overhead.
"""

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoTokenizer

WORKTREE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKTREE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKTREE_ROOT))

import nimble.training.schema_train as st
from nimble.evaluation.evaluate_pilot import adapt_input
from nimble.multimodal import (
    IMAGE_PAD,
    MMCandidateCollator,
    build_mm_runtime_row,
    cyclic_shifts_for,
    encode_mm_one,
    image_for,
)
from nimble.scoring.parallel_schema import choice_key
from nimble.training.model_loading import load_base
from playground_web.api_schema import count_output_tokens, format_systemone_answers, render_flat

MAX_CHOICES = 26
MAX_LENGTH = 2048


@dataclass
class NimbleMethodSpec:
    name: str
    label: str
    group: str
    base_model: str
    revision: str
    adapter_dir: Path
    multimodal: bool
    prompt_format: str            # "sandwich_v3" or "v1"
    fp32_head: bool
    locked_temperature: float = 1.0
    tta_locked_temperature: float = 1.0
    test_acc_note: str = ""


def _clean_text(value):
    if isinstance(value, str):
        return value if value.strip() else None
    flat = render_flat(value).strip() if value is not None else ""
    return flat or None


def sanitize_question(q_dict: dict, qid: str) -> dict:
    kind, crit = q_dict["type"], q_dict.get("criteria")
    instructions = _clean_text(q_dict.get("instructions")) or qid
    if kind == "choice":
        criteria = {
            k: (_clean_text(v) or ("广告所属行业不在以上候选类目中" if k == "以上都不是" else k))
            for k, v in (crit or {}).items()
        }
    elif kind == "noul":
        crit = crit or {}
        criteria = {k: _clean_text(crit.get(k)) or k for k in ("true", "false")}
    elif kind == "score":
        criteria = [_clean_text(v) or str(i) for i, v in enumerate(crit or [])]
    else:
        raise ValueError(f"Unsupported question type: {kind}")
    return {"type": kind, "instructions": instructions, "criteria": criteria}


class SharedBasePool:
    """Holds one loaded Qwen3.5 base checkpoint and hot-swaps multiple LoRA adapters."""

    def __init__(self, base_model: str, revision: str):
        self.base_model = base_model
        self.revision = revision
        offline = True
        self.tok = AutoTokenizer.from_pretrained(base_model, revision=revision, local_files_only=offline)
        self.tok.padding_side = "left"
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self.base = load_base(base_model, revision)
        self.peft_model: PeftModel | None = None
        self.loaded_adapters: set[str] = set()
        self.collator = MMCandidateCollator(self.tok.pad_token_id)

    def register_adapter(self, adapter_key: str, adapter_dir: Path):
        if adapter_key in self.loaded_adapters:
            return
        dev = next(self.base.parameters()).device
        if self.peft_model is None:
            self.peft_model = PeftModel.from_pretrained(
                self.base, str(adapter_dir), adapter_name=adapter_key, is_trainable=False
            ).to(dev).eval()
        else:
            self.peft_model.load_adapter(str(adapter_dir), adapter_name=adapter_key, is_trainable=False)
            self.peft_model.to(dev).eval()
        self.loaded_adapters.add(adapter_key)


class NimbleMethodPredictor:
    """Serves a single ``NimbleMethodSpec`` backed by a ``SharedBasePool``."""

    def __init__(self, spec: NimbleMethodSpec, pool: SharedBasePool):
        self.spec = spec
        self.pool = pool
        self.tok = pool.tok
        self.adapter_key = spec.name.replace(".", "_").replace("-", "_")
        pool.register_adapter(self.adapter_key, spec.adapter_dir)

    def _prepare_decisions(self, req):
        state = req.state
        image_src = None
        if isinstance(state, dict) and "image" in state:
            image_src = state["image"]
            state = {k: v for k, v in state.items() if k != "image"}
        # Pure text methods gracefully ignore state.image during side-by-side comparison!
        image = image_for(image_src) if (self.spec.multimodal and image_src is not None) else None
        decisions = []
        for qid, q in req.questions.items():
            sq = sanitize_question(q.model_dump(), qid)
            if sq["type"] == "choice" and len(sq["criteria"]) > MAX_CHOICES:
                raise ValueError(f"Nimble supports up to {MAX_CHOICES} choices; '{qid}' has {len(sq['criteria'])}")
            if isinstance(state, dict) and not state:
                _, schema = adapt_input({"state": {"_empty": "1"}, "questions": {"decision": sq}})
                context = ""
            elif isinstance(state, str) and not state.strip():
                _, schema = adapt_input({"state": {"_empty": "1"}, "questions": {"decision": sq}})
                context = ""
            else:
                context, schema = adapt_input({"state": state, "questions": {"decision": sq}})
            decisions.append((qid, sq, context, schema, image))
        return decisions

    def _forward_logits(self, row: dict) -> torch.Tensor:
        model = self.pool.peft_model
        model.set_adapter(self.adapter_key)
        device = next(model.parameters()).device
        batch = {k: v.to(device) for k, v in self.pool.collator([row]).items()}
        saved = (
            torch.backends.cuda.flash_sdp_enabled(),
            torch.backends.cuda.mem_efficient_sdp_enabled(),
            torch.backends.cuda.math_sdp_enabled(),
        )
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
        try:
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                logits = st.candidate_logits(model, batch, fp32_candidate_head=self.spec.fp32_head)[0].cpu()
        finally:
            torch.backends.cuda.enable_flash_sdp(saved[0])
            torch.backends.cuda.enable_mem_efficient_sdp(saved[1])
            torch.backends.cuda.enable_math_sdp(saved[2])
        return logits[:len(row["choices"])].double()

    def answer(self, req, name: str) -> dict:
        start = time.perf_counter()
        decisions = self._prepare_decisions(req)
        use_cal = bool(getattr(req, "calibrated", True))
        tta_views = max(1, int(getattr(req, "tta_views", 1) or 1))

        probs_list, meta_list = [], []
        total_input_tokens = 0
        applied_temps = []

        for qid, sq, context, schema, image in decisions:
            enc0 = encode_mm_one(
                self.tok, context, schema, image=image,
                prompt_format=self.spec.prompt_format,
                row_id=qid, kind=sq["type"], max_length=MAX_LENGTH,
            )
            row0 = build_mm_runtime_row(enc0, qid, qid, sq["type"])
            total_input_tokens += len(row0["input_ids"])
            base_logits = self._forward_logits(row0)
            keys0 = [choice_key(c) for c in row0["choices"]]

            if tta_views > 1 and sq["type"] == "choice" and len(keys0) > 1:
                shifts = cyclic_shifts_for(len(keys0), tta_views)
                acc = {k: [float(base_logits[i] - base_logits.mean())] for i, k in enumerate(keys0)}
                for sh in shifts:
                    if sh == 0:
                        continue
                    enc_s = encode_mm_one(
                        self.tok, context, schema, image=image,
                        prompt_format=self.spec.prompt_format,
                        row_id=qid, kind=sq["type"], cyclic_shift=sh, max_length=MAX_LENGTH,
                    )
                    row_s = build_mm_runtime_row(enc_s, qid, qid, sq["type"])
                    total_input_tokens += len(row_s["input_ids"])
                    l_s = self._forward_logits(row_s)
                    m_s = l_s.mean()
                    for idx_s, c_val in enumerate(row_s["choices"]):
                        acc[choice_key(c_val)].append(float(l_s[idx_s] - m_s))
                merged = torch.tensor([sum(acc[k]) / len(acc[k]) for k in keys0], dtype=torch.float64)
                temp = self.spec.tta_locked_temperature if use_cal else 1.0
            else:
                merged = base_logits
                temp = self.spec.locked_temperature if use_cal else 1.0

            applied_temps.append(round(float(temp), 4))
            scaled = merged / float(temp)
            probs = scaled.softmax(-1).tolist()
            probs_list.append(probs)
            m = {"id": qid, "type": sq["type"], "keys": keys0}
            if sq["type"] == "score":
                m["legend"] = {str(i): (_clean_text(v) or str(i)) for i, v in enumerate(sq["criteria"])}
            meta_list.append(m)

        answers = format_systemone_answers(probs_list, meta_list)
        return {
            "model": name,
            "method_meta": {
                "label": self.spec.label,
                "prompt_format": self.spec.prompt_format,
                "fp32_head": self.spec.fp32_head,
                "multimodal": self.spec.multimodal,
                "temperature_applied": applied_temps[0] if len(applied_temps) == 1 else applied_temps,
                "tta_views_used": tta_views,
                "benchmark_note": self.spec.test_acc_note,
            },
            "answers": answers,
            "usage": {"input_tokens": total_input_tokens, "output_tokens": count_output_tokens(self.tok, answers)},
            "evaluation_time_ms": round((time.perf_counter() - start) * 1000, 1),
        }

    def prompt(self, req, name: str) -> dict:
        decisions = self._prepare_decisions(req)
        tok = self.tok
        groups, table, rendered = [], [], []
        rendered_state_all = ""

        for qid, sq, context, schema, image in decisions:
            enc = encode_mm_one(
                tok, context, schema, image=image,
                prompt_format=self.spec.prompt_format,
                row_id=qid, kind=sq["type"], max_length=MAX_LENGTH,
            )
            ids = enc["prompt_token_ids"]
            mm = enc["mm"]
            seg = f"question {qid} ({self.spec.prompt_format})"
            rendered_state_all = context or "(empty text context — visual only)"
            image_info = {"image_tokens": sum(mm), "grid_patches_thw": list(image[1])} if image is not None else None
            opts = (
                [f"{k}: {v}" if v != k else k for k, v in sq["criteria"].items()]
                if sq["type"] == "choice"
                else ([f"false: {sq['criteria']['false']}", f"true: {sq['criteria']['true']}"]
                      if sq["type"] == "noul" else [f"{i}: {v}" for i, v in enumerate(sq["criteria"])])
            )
            rendered.append({"id": qid, "type": sq["type"], "instructions": sq["instructions"],
                             "options": opts, **({"image": image_info} if image_info else {})})

            i = 0
            while i < len(ids):
                piece = tok.convert_ids_to_tokens(ids[i])
                if mm[i]:
                    j = i
                    while j < len(ids) and mm[j]:
                        j += 1
                    groups.append({"segment": seg, "kind": "image", "text": piece, "count": j - i})
                elif piece.startswith("<|") and piece.endswith("|>"):
                    j = i + 1
                    groups.append({"segment": seg, "kind": "special", "text": piece, "count": 1})
                else:
                    j = i + 1
                    while (j < len(ids) and not mm[j]
                           and not (tok.convert_ids_to_tokens(ids[j]).startswith("<|")
                                    and tok.convert_ids_to_tokens(ids[j]).endswith("|>"))):
                        j += 1
                    groups.append({"segment": seg, "kind": "text", "text": tok.decode(ids[i:j], skip_special_tokens=False), "count": j - i})
                for k in range(i, j):
                    table.append({"i": len(table), "id": ids[k], "piece": tok.convert_ids_to_tokens(ids[k]), "segment": seg, "opt": ""})
                i = j

        return {
            "model": name,
            "summary": {
                "total_tokens": len(table),
                "state_total": None,
                "image_tokens": sum(g["count"] for g in groups if g["kind"] == "image"),
                "questions": len(rendered),
                "prompt_format": self.spec.prompt_format,
                "fp32_head": self.spec.fp32_head,
            },
            "rendered": {
                "state": rendered_state_all,
                "image": rendered[0].get("image") if rendered else None,
                "questions": [{k: v for k, v in q.items() if k != "image"} for q in rendered],
            },
            "segments": groups,
            "tokens": table,
        }
