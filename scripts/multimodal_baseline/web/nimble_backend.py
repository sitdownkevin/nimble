"""Nimble backend for the playground.

Two checkpoints on Qwen3.5-0.8B-Base, both trained by nimble's own trainer:
  nimble-0.8b       multimodal: the advertisement image is in nimble's prompt (scripts/nimble_mm.py), trained by
                    scripts/nimble_mm_train.py (runs/nimble-0.8b-mm-700). Requests must carry state.image.
  nimble-0.8b-text  text only: nimble's own prompt with the text fields (runs/nimble-0.8b-text-700).

Each question of a request is one nimble decision over the request's state, scored with nimble's candidate-logit scorer.
Answers use kev's System One answer format (kev.api.to_answers), so the Response panel and /v1/systemone look the same
for every model. The Prompt view shows the prompt the model actually reads.
"""
import sys, time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
NIMBLE = Path("/root/projects/nimble")
KEV = Path("/root/projects/kev")
SCRIPTS = ROOT / "scripts"
for p in (NIMBLE, KEV, SCRIPTS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import nimble.training.schema_train as st
from nimble.evaluation.evaluate_pilot import adapt_input
from nimble.scoring.parallel_schema import choice_key
from nimble.training.schema_data import as_scoring, encode_scoring, runtime_record
from kev.api import output_tokens, render, to_answers
import nimble_mm as M

NIMBLE_MM_NAME = "nimble-0.8b"
NIMBLE_TEXT_NAME = "nimble-0.8b-text"
MM_ADAPTER = ROOT / "runs/nimble-0.8b-mm-700"
TEXT_ADAPTER = ROOT / "runs/nimble-0.8b-text-700"
NIMBLE_BASE = "Qwen/Qwen3.5-0.8B-Base"
NIMBLE_REVISION = "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"
MAX_CHOICES = st.MAX_CHOICES          # one letter code per option (A-Z)
MAX_LENGTH = 2048                     # the training prompt limit


def text(value):
    """kev's rendered text of a value, or None when it is empty. Strings are kept exactly as they are."""
    if isinstance(value, str):
        return value if value.strip() else None
    flat = render(value).strip() if value is not None else ""
    return flat or None


def sanitize(q, qid):
    """A kev question (dict) -> nimble question: instructions text and criteria with a nonempty string per option."""
    kind, crit = q["type"], q.get("criteria")
    instructions = text(q.get("instructions")) or qid
    if kind == "choice":
        criteria = {k: text(v) or k for k, v in (crit or {}).items()}
    elif kind == "noul":
        crit = crit or {}
        criteria = {k: text(crit.get(k)) or k for k in ("true", "false")}
    elif kind == "score":
        criteria = [text(v) or str(i) for i, v in enumerate(crit or [])]
    else:
        raise ValueError(f"unsupported question type {kind}")
    return {"type": kind, "instructions": instructions, "criteria": criteria}


class NimblePredictor:
    def __init__(self, adapter, multimodal):
        self.multimodal = multimodal
        self.tok = AutoTokenizer.from_pretrained(NIMBLE_BASE, revision=NIMBLE_REVISION)
        self.tok.padding_side = "left"
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        base = st.load_base(NIMBLE_BASE, NIMBLE_REVISION)
        self.model = PeftModel.from_pretrained(base, str(adapter)).eval()
        self.device = next(self.model.parameters()).device
        self.collator = M.MMCollator(self.tok.pad_token_id) if multimodal else st.CandidateCollator(self.tok.pad_token_id)
        self.logits_fn = M.candidate_logits_mm if multimodal else st.candidate_logits

    # --- request -> nimble decisions -------------------------------------------------------------------------------
    def _decisions(self, req):
        """-> [(question id, sanitized question, state without image, image or None)] for every question of the request."""
        state, image_src = req.state, None
        if isinstance(state, dict) and "image" in state:
            image_src = state["image"]
            state = {k: v for k, v in state.items() if k != "image"}
        if not self.multimodal and image_src is not None:
            raise ValueError(f"{NIMBLE_TEXT_NAME} reads text only: remove state.image, or use {NIMBLE_MM_NAME}")
        image = M.image_for(image_src) if image_src is not None else None
        out = []
        for qid, q in req.questions.items():
            sq = sanitize(q.model_dump(), qid)
            if sq["type"] == "choice" and len(sq["criteria"]) > MAX_CHOICES:
                raise ValueError(f"nimble scores at most {MAX_CHOICES} options per question; {qid} has {len(sq['criteria'])}")
            out.append((qid, sq, state, image))
        return out

    def _rows(self, decisions):
        """-> one runtime row per decision (nimble's own row format; the multimodal rows also carry the image inputs)."""
        rows, n_tokens = [], []
        for qid, sq, state, image in decisions:
            if self.multimodal:
                context, schema = adapt_input({"state": state, "questions": {"decision": sq}})
                enc = M.encode_one(self.tok, context, schema, image, row_id=qid)
                row = M.runtime_row(enc, qid, qid, sq["type"])
                n_tokens.append(len(enc["prompt_token_ids"]))
            else:
                target = next(iter(sq["criteria"])) if sq["type"] == "choice" else (False if sq["type"] == "noul" else 0)
                raw = {"id": qid, "family": qid, "source_family": qid, "provenance": {"source_id": qid},
                       "input": {"state": state, "questions": {"decision": sq}}, "reference": {"target": target}}
                enc = encode_scoring(as_scoring(raw, False), self.tok, MAX_LENGTH)
                row = runtime_record(enc, raw)
                n_tokens.append(len(enc["prompt_token_ids"]))
            rows.append(row)
        return rows, n_tokens

    @torch.no_grad()
    def _decide(self, row):
        batch = {k: v.to(self.device) for k, v in self.collator([row]).items()}
        # kev's LocalPredictor turns off flash / memory-efficient SDPA process-wide. nimble trained and evaluated with
        # PyTorch's defaults (all SDPA kernels on), and bf16 attention rounds differently per kernel, so restore those
        # defaults for nimble's forward pass only (requests run one at a time under the server's lock).
        saved = (torch.backends.cuda.flash_sdp_enabled(), torch.backends.cuda.mem_efficient_sdp_enabled(), torch.backends.cuda.math_sdp_enabled())
        torch.backends.cuda.enable_flash_sdp(True); torch.backends.cuda.enable_mem_efficient_sdp(True); torch.backends.cuda.enable_math_sdp(True)
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16):   # the same precision as nimble's evaluation
                logits = self.logits_fn(self.model, batch)[0].cpu()
        finally:
            torch.backends.cuda.enable_flash_sdp(saved[0]); torch.backends.cuda.enable_mem_efficient_sdp(saved[1]); torch.backends.cuda.enable_math_sdp(saved[2])
        return st.decision_result(row, logits)

    # --- public API ------------------------------------------------------------------------------------------------
    def answer(self, req, name):
        start = time.perf_counter()
        decisions = self._decisions(req)
        rows, n_tokens = self._rows(decisions)
        probs, meta = [], []
        for (qid, sq, _, _), row in zip(decisions, rows):
            res = self._decide(row)
            keys = [choice_key(c) for c in row["choices"]]
            probs.append([res["probabilities"][k] for k in keys])
            m = {"id": qid, "type": sq["type"], "keys": keys}
            if sq["type"] == "score":
                m["legend"] = {str(i): (text(v) or str(i)) for i, v in enumerate(sq["criteria"])}
            meta.append(m)
        answers = to_answers(probs, meta)
        return {"model": name, "answers": answers,
                "usage": {"input_tokens": sum(n_tokens), "output_tokens": output_tokens(self.tok, answers)},
                "evaluation_time_ms": round((time.perf_counter() - start) * 1000, 1)}

    def prompt(self, req, name):
        """The token sequence nimble's tokenizer produced for each question: rendered text, segments, token table."""
        decisions = self._decisions(req)
        rows, _ = self._rows(decisions)
        tok = self.tok
        img_pad = tok.convert_tokens_to_ids(M.IMAGE_PAD) if self.multimodal else None
        groups, table, rendered = [], [], []
        for (qid, sq, state, image), row in zip(decisions, rows):
            seg = f"question {qid}"
            ids = row["input_ids"]
            mm = row.get("mm") or [0] * len(ids)
            if self.multimodal:
                context, _ = adapt_input({"state": state, "questions": {"decision": sq}})
                n_img = sum(mm)
                rendered_state = context
                image_info = {"image_tokens": n_img, "grid_patches_thw": list(image[1])} if image is not None else None
            else:
                context = as_scoring({"id": qid, "family": qid, "provenance": {"source_id": qid},
                                      "input": {"state": state, "questions": {"decision": sq}},
                                      "reference": {"target": next(iter(sq["criteria"])) if sq["type"] == "choice" else (False if sq["type"] == "noul" else 0)}}, False)["context"]
                rendered_state = context
                image_info = None
            rendered.append({"id": qid, "type": sq["type"], "instructions": sq["instructions"], "options": self._option_lines(sq)})
            rendered_state_text = rendered_state
            i = 0
            while i < len(ids):
                piece = tok.convert_ids_to_tokens(ids[i])
                if mm[i]:                                   # image placeholder run
                    j = i
                    while j < len(ids) and mm[j]:
                        j += 1
                    groups.append({"segment": seg, "kind": "image", "text": piece, "count": j - i})
                elif piece.startswith("<|") and piece.endswith("|>"):
                    j = i + 1
                    groups.append({"segment": seg, "kind": "special", "text": piece, "count": 1})
                else:
                    j = i + 1
                    while j < len(ids) and not mm[j] and not (tok.convert_ids_to_tokens(ids[j]).startswith("<|") and tok.convert_ids_to_tokens(ids[j]).endswith("|>")):
                        j += 1
                    groups.append({"segment": seg, "kind": "text", "text": tok.decode(ids[i:j], skip_special_tokens=False), "count": j - i})
                for k in range(i, j):
                    table.append({"i": len(table), "id": ids[k], "piece": tok.convert_ids_to_tokens(ids[k]), "segment": seg, "opt": ""})
                i = j
            if image_info:
                rendered[-1]["image"] = image_info
        rendered_state_all = rendered_state_text if rendered_state_text else ""
        return {"model": name,
                "summary": {"total_tokens": len(table), "state_total": None,
                            "image_tokens": sum(1 for g in groups if g["kind"] == "image" for _ in range(g["count"])),
                            "questions": len(rendered)},
                "rendered": {"state": rendered_state_all, "image": rendered[0].get("image") if rendered else None,
                             "questions": [{k: v for k, v in q.items() if k != "image"} for q in rendered]},
                "segments": groups, "tokens": table}

    @staticmethod
    def _option_lines(sq):
        if sq["type"] == "choice":
            return [f"{k}: {v}" if v != k else k for k, v in sq["criteria"].items()]
        if sq["type"] == "noul":
            return [f"false: {sq['criteria']['false']}", f"true: {sq['criteria']['true']}"]
        return [f"{i}: {v}" for i, v in enumerate(sq["criteria"])]
