#!/usr/bin/env python3
"""Multi-Method Nimble & Kev Interactive Web Playground.

Lets users visually compare Legacy v1, Multimodal SOTA (sandwich_v3 + FP32 + Merger/Vision LoRA + EMA + Locked-T),
Text-Only controls, Raw-Ad pure pixel perception, and Kev pointer-head models side-by-side in real time.
"""

import argparse
import base64
import io
import json
import os
import random
import sys
import threading
import time
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from PIL import Image

WORKTREE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKTREE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKTREE_ROOT))

from playground_web.api_schema import CompareAllRequest, SystemOneRequest
from playground_web.predictors import NimbleMethodPredictor, NimbleMethodSpec, SharedBasePool

MIME = {"jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp", "gif": "image/gif", "bmp": "image/bmp"}
LOCK = threading.Lock()
PREDICTORS: dict[str, object] = {}
MODEL_CATALOG: list[dict] = []
KEV_HELPERS = None

DATA_DIR = Path(os.environ.get("JEV_DATA_DIR", "/root/projects/jev-real-ad-1000/kev_data"))
LEGACY_RUNS_DIR = Path(os.environ.get("LEGACY_RUNS_DIR", "/root/projects/jev-real-ad-1000/runs"))
SOTA_RUNS_DIR = Path(os.environ.get("SOTA_RUNS_DIR", "/data/nvme0/nimble_mm_exp/runs"))
KEV_ROOT = Path(os.environ.get("KEV_ROOT", "/root/projects/kev"))

_TEST_SAMPLES = None
_DISAGREEMENT_IDS: list[str] = []

app = FastAPI(title="Nimble & Kev Multimodal SOTA Playground")


def image_data_uri(raw: bytes) -> str:
    try:
        with Image.open(io.BytesIO(raw)) as im:
            fmt = (im.format or "").lower()
            im.load()
    except Exception:
        raise HTTPException(400, "Unreadable image")
    if fmt not in MIME:
        raise HTTPException(400, f"Unsupported image format: {fmt or 'unknown'}")
    return f"data:{MIME[fmt]};base64," + base64.b64encode(raw).decode()


def resolve_model(model: str) -> str:
    if model in ("nimble-sota", "default", "kev-latest"):
        for cand in ("nimble-4b-mm-sota", "nimble-0.8b-mm-sota"):
            if cand in PREDICTORS:
                return cand
        return next(iter(PREDICTORS))
    if model in PREDICTORS:
        return model
    raise HTTPException(400, f"Unknown model {model!r}; available: {list(PREDICTORS)}")


def load_test_records() -> list[dict]:
    global _TEST_SAMPLES, _DISAGREEMENT_IDS
    if _TEST_SAMPLES is not None:
        return _TEST_SAMPLES
    out = []
    test_jsonl = DATA_DIR / "test_pos_start.jsonl"
    if not test_jsonl.exists():
        _TEST_SAMPLES = []
        return _TEST_SAMPLES

    # Check precomputed predictions to tag disagreement showcase samples (Legacy wrong -> SOTA right!)
    legacy_correct = {}
    sota08_correct = {}
    sota4b_correct = {}
    leg_p = LEGACY_RUNS_DIR / "nimble-0.8b-mm-700/eval/test-final.json"
    if leg_p.exists():
        legacy_correct = {r["id"]: bool(r["correct"]) for r in json.loads(leg_p.read_text())["rows"]}
    s4_p = SOTA_RUNS_DIR / "mm11_4b_v3_merger_lora/eval/test_final_ema_tta5.json"
    if s4_p.exists():
        sota4b_correct = {r["id"]: bool(r["correct"]) for r in json.loads(s4_p.read_text())["rows"]}
    s08_p = SOTA_RUNS_DIR / "mm01_08b_v3_frozen/eval/test_final_ema.json"
    if s08_p.exists():
        sota08_correct = {r["id"]: bool(r["correct"]) for r in json.loads(s08_p.read_text())["rows"]}

    with test_jsonl.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            state = dict(r["state"])
            image_rel = state.pop("image")
            q = next(iter(r["questions"].values()))
            rid = r["id"]
            has_copy = bool(state.get("title") or state.get("body"))
            is_disagree = (legacy_correct.get(rid) is False) and (sota08_correct.get(rid) or sota4b_correct.get(rid))
            if is_disagree:
                _DISAGREEMENT_IDS.append(rid)
            out.append({
                "id": rid,
                "gold": q["label"],
                "has_copy": has_copy,
                "is_disagree": is_disagree,
                "state": state,
                "image_rel": image_rel,
                "questions": {"industry": {"type": "choice", "instructions": q["instructions"], "criteria": q["criteria"]}},
            })
    _TEST_SAMPLES = out
    return _TEST_SAMPLES


def _run_kev_answer(kev_pred, req: SystemOneRequest, name: str) -> dict:
    start = time.perf_counter()
    kh = KEV_HELPERS
    kev_req = kh["SystemOneRequest"](state=req.state, model=name, questions={k: v.model_dump() for k, v in req.questions.items()})
    rec, meta = kh["to_record"](kev_req)
    ctx = kh["CONTEXT"]
    enc = kev_pred.model.encode(kev_pred.tok, rec, max_state=ctx["max_state"], max_branch=ctx["max_branch"], strict=True)
    logits = kev_pred.model.forward(enc)
    probs = [torch.softmax(z.float(), -1).cpu().tolist() for z in logits]
    answers = kh["to_answers"](probs, meta)
    return {
        "model": name,
        "method_meta": {"label": f"Kev Pointer-Head ({name})", "prompt_format": "kev_fim_pointer", "fp32_head": True,
                        "multimodal": True, "temperature_applied": 1.0, "tta_views_used": 1},
        "answers": answers,
        "usage": {"input_tokens": len(enc["ids"]), "output_tokens": kh["output_tokens"](kev_pred.tok, answers)},
        "evaluation_time_ms": round((time.perf_counter() - start) * 1000, 1),
    }


def _run_kev_prompt(kev_pred, req: SystemOneRequest, name: str) -> dict:
    kh = KEV_HELPERS
    kev_req = kh["SystemOneRequest"](state=req.state, model=name, questions={k: v.model_dump() for k, v in req.questions.items()})
    rec, meta = kh["to_record"](kev_req)
    ctx = kh["CONTEXT"]
    enc = kev_pred.model.encode(kev_pred.tok, rec, max_state=ctx["max_state"], max_branch=ctx["max_branch"], strict=True)
    tok = kev_pred.tok
    ids, seg, opt = enc["ids"], enc["seg"], enc["opt"]
    img_id = tok.convert_tokens_to_ids("<|image_pad|>")
    sp_ids = {tok.convert_tokens_to_ids(t) for t in kh["SPECIAL"] + ["<|vision_start|>", "<|vision_end|>"]}
    qids = [m["id"] for m in meta]

    def kind(t):
        return "image" if t == img_id else ("special" if t in sp_ids else "text")

    def seg_name(s):
        return "state" if s == 0 else f"question {qids[s - 1]}"

    def opt_label(o):
        return "" if o == -1 else ("decide" if o == -2 else f"option {o}")

    groups, i = [], 0
    while i < len(ids):
        k, s, j = kind(ids[i]), seg[i], i + 1
        if k == "special":
            text = tok.convert_ids_to_tokens(ids[i])
        elif k == "image":
            while j < len(ids) and seg[j] == s and kind(ids[j]) == k:
                j += 1
            text = tok.convert_ids_to_tokens(ids[i])
        else:
            while j < len(ids) and seg[j] == s and kind(ids[j]) == k:
                j += 1
            text = tok.decode(ids[i:j], skip_special_tokens=False)
        groups.append({"segment": seg_name(s), "kind": k, "text": text, "count": j - i})
        i = j

    rows = [{"i": n, "id": t, "piece": tok.convert_ids_to_tokens(t), "segment": seg_name(seg[n]), "opt": opt_label(opt[n])}
            for n, t in enumerate(ids)]
    n_img = sum(1 for t in ids if t == img_id)
    image = {"image_tokens": n_img, "grid_patches_thw": list(kh["image_inputs"](rec["image"])[1])} if rec.get("image") else None
    return {
        "model": name,
        "summary": {"total_tokens": len(ids), "state_total": sum(1 for s in seg if s == 0), "image_tokens": n_img,
                    "questions": len(meta), "prompt_format": "kev_fim_pointer", "fp32_head": True},
        "rendered": {"state": rec["state"], "image": image,
                     "questions": [{"id": m["id"], "type": m["type"], "instructions": q["instr"], "options": q["options"]}
                                   for m, q in zip(meta, rec["questions"])]},
        "segments": groups,
        "tokens": rows,
    }


def evaluate_single(req: SystemOneRequest, name: str) -> dict:
    pred = PREDICTORS[name]
    with LOCK, torch.no_grad():
        try:
            if isinstance(pred, NimbleMethodPredictor):
                return pred.answer(req, name)
            return _run_kev_answer(pred, req, name)
        except ValueError as e:
            raise HTTPException(422, str(e))


@app.post("/v1/systemone")
def systemone(req: SystemOneRequest):
    return evaluate_single(req, resolve_model(req.model))


@app.post("/api/compare")
def compare_all(req: CompareAllRequest):
    target_models = req.models or [m["id"] for m in MODEL_CATALOG]
    results = []
    t0 = time.perf_counter()
    for mid in target_models:
        if mid not in PREDICTORS:
            continue
        sub_req = SystemOneRequest(
            state=req.state, model=mid, questions=req.questions,
            calibrated=req.calibrated, tta_views=req.tta_views,
        )
        try:
            res = evaluate_single(sub_req, mid)
            results.append({"model": mid, "ok": True, "result": res})
        except HTTPException as e:
            results.append({"model": mid, "ok": False, "error": e.detail})
    return {
        "total_wall_ms": round((time.perf_counter() - t0) * 1000, 1),
        "calibrated": req.calibrated,
        "tta_views": req.tta_views,
        "results": results,
    }


@app.post("/api/prompt")
def prompt_endpoint(req: SystemOneRequest):
    name = resolve_model(req.model)
    pred = PREDICTORS[name]
    with LOCK, torch.no_grad():
        try:
            if isinstance(pred, NimbleMethodPredictor):
                return pred.prompt(req, name)
            return _run_kev_prompt(pred, req, name)
        except ValueError as e:
            raise HTTPException(422, str(e))


@app.get("/v1/models")
def list_models():
    return list(PREDICTORS.keys())


@app.get("/api/catalog")
def get_catalog():
    recs = load_test_records()
    return {
        "models": MODEL_CATALOG,
        "sample_counts": {
            "all": len(recs),
            "disagree": len(_DISAGREEMENT_IDS),
            "poster_only": sum(1 for r in recs if not r["has_copy"]),
        },
    }


@app.get("/api/sample/random")
def random_sample(mode: str = "all"):
    recs = load_test_records()
    if not recs:
        raise HTTPException(404, "No test samples loaded")
    if mode == "disagree":
        pool = [r for r in recs if r["is_disagree"]] or recs
    elif mode == "poster_only":
        pool = [r for r in recs if not r["has_copy"]] or recs
    else:
        pool = recs
    rec = random.choice(pool)
    return {
        "id": rec["id"],
        "gold": rec["gold"],
        "has_copy": rec["has_copy"],
        "is_disagree": rec["is_disagree"],
        "state": rec["state"],
        "questions": rec["questions"],
        "image": image_data_uri((DATA_DIR / rec["image_rel"]).read_bytes()),
    }


STATIC_DIR = Path(__file__).resolve().parent / "static"


@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(
        (STATIC_DIR / "index.html").read_text(encoding="utf-8"),
        headers={"Cache-Control": "no-cache"},
    )


@app.get("/static/{filename}")
def static_asset(filename: str):
    target = (STATIC_DIR / filename).resolve()
    if not target.is_file() or STATIC_DIR.resolve() not in target.parents:
        raise HTTPException(404, "Static asset not found")
    return FileResponse(target, headers={"Cache-Control": "no-cache"})


def register_all_models():
    base_08b = ("Qwen/Qwen3.5-0.8B-Base", "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68")
    base_4b = ("Qwen/Qwen3.5-4B-Base", "1001bb4d826a52d1f399e183466143f4da7b741b")

    specs = [
        NimbleMethodSpec(
            name="nimble-4b-mm-sota",
            label="🏆 Nimble-4B 多模态 SOTA (sandwich_v3 + Merger-LoRA)",
            group="4B SOTA",
            base_model=base_4b[0], revision=base_4b[1],
            adapter_dir=SOTA_RUNS_DIR / "mm11_4b_v3_merger_lora/ema_adapter",
            multimodal=True, prompt_format="sandwich_v3", fp32_head=True,
            locked_temperature=1.1997, tta_locked_temperature=1.1213,
            test_acc_note="Test 86.67% (TTA5) / Val 87.33%",
        ),
        NimbleMethodSpec(
            name="nimble-4b-mm-lora-all",
            label="🏆 Nimble-4B 多模态全视觉LoRA (sandwich_v3 + LoRA-All)",
            group="4B SOTA",
            base_model=base_4b[0], revision=base_4b[1],
            adapter_dir=SOTA_RUNS_DIR / "mm12_4b_v3_lora_all/ema_adapter",
            multimodal=True, prompt_format="sandwich_v3", fp32_head=True,
            locked_temperature=1.1431, tta_locked_temperature=1.1093,
            test_acc_note="单步 Test 84.00% (最高) / ECE 0.043",
        ),
        NimbleMethodSpec(
            name="nimble-4b-mm-frozen",
            label="Nimble-4B 多模态冻结视觉塔 (sandwich_v3 + Frozen)",
            group="4B SOTA",
            base_model=base_4b[0], revision=base_4b[1],
            adapter_dir=SOTA_RUNS_DIR / "mm10_4b_v3_frozen/ema_adapter",
            multimodal=True, prompt_format="sandwich_v3", fp32_head=True,
            locked_temperature=1.2415, tta_locked_temperature=1.1950,
            test_acc_note="最佳概率校准 NLL=0.4080 / Test 86.00% (TTA5)",
        ),
        NimbleMethodSpec(
            name="nimble-4b-text-sota",
            label="Nimble-4B 纯文本对照 (sandwich_v3, 忽略图片)",
            group="4B Control",
            base_model=base_4b[0], revision=base_4b[1],
            adapter_dir=SOTA_RUNS_DIR / "mm13_4b_text_v3_sota/ema_adapter",
            multimodal=False, prompt_format="sandwich_v3", fp32_head=True,
            locked_temperature=1.3376, tta_locked_temperature=1.2994,
            test_acc_note="纯文本对照 Test 83.33% (无视觉能力)",
        ),
        NimbleMethodSpec(
            name="nimble-0.8b-mm-sota",
            label="🏆 Nimble-0.8B 多模态 SOTA (sandwich_v3 + Frozen)",
            group="0.8B SOTA",
            base_model=base_08b[0], revision=base_08b[1],
            adapter_dir=SOTA_RUNS_DIR / "mm01_08b_v3_frozen/ema_adapter",
            multimodal=True, prompt_format="sandwich_v3", fp32_head=True,
            locked_temperature=1.4825, tta_locked_temperature=1.4445,
            test_acc_note="0.8B 单步冠军 Test 79.33% / NLL 0.794",
        ),
        NimbleMethodSpec(
            name="nimble-0.8b-rawad-mm",
            label="Nimble-0.8B 纯原图训练模型 (无OCR训练 · 专测纯看图)",
            group="0.8B Raw-Ad",
            base_model=base_08b[0], revision=base_08b[1],
            adapter_dir=SOTA_RUNS_DIR / "mm08_08b_rawad_mm_frozen/ema_adapter",
            multimodal=True, prompt_format="sandwich_v3", fp32_head=True,
            locked_temperature=1.4238, tta_locked_temperature=1.3728,
            test_acc_note="无OCR纯看图 Test 76.00% (+13.33pp vs 纯文本)",
        ),
        NimbleMethodSpec(
            name="nimble-0.8b-legacy-v1",
            label="旧版基线 Nimble-0.8B 多模态 (v1 双重JSON + BF16头)",
            group="Legacy v1",
            base_model=base_08b[0], revision=base_08b[1],
            adapter_dir=LEGACY_RUNS_DIR / "nimble-0.8b-mm-700",
            multimodal=True, prompt_format="v1", fp32_head=False,
            locked_temperature=1.0, tta_locked_temperature=1.0,
            test_acc_note="旧基线 Test 76.00% / NLL 1.542 (过自信)",
        ),
        NimbleMethodSpec(
            name="nimble-0.8b-text-legacy-v1",
            label="旧版基线 Nimble-0.8B 纯文本 (v1 JSON, 忽略图片)",
            group="Legacy v1",
            base_model=base_08b[0], revision=base_08b[1],
            adapter_dir=LEGACY_RUNS_DIR / "nimble-0.8b-text-700",
            multimodal=False, prompt_format="v1", fp32_head=False,
            locked_temperature=1.0, tta_locked_temperature=1.0,
            test_acc_note="旧纯文本基线 Test 77.33% / NLL 1.234",
        ),
    ]

    pools: dict[tuple[str, str], SharedBasePool] = {}
    for sp in specs:
        if not sp.adapter_dir.exists():
            print(f"[Playground] Skipping {sp.name} (missing {sp.adapter_dir})", flush=True)
            continue
        key = (sp.base_model, sp.revision)
        if key not in pools:
            print(f"[Playground] Loading shared base backbone {sp.base_model}@{sp.revision[:8]}...", flush=True)
            pools[key] = SharedBasePool(sp.base_model, sp.revision)
        PREDICTORS[sp.name] = NimbleMethodPredictor(sp, pools[key])
        MODEL_CATALOG.append({"id": sp.name, "title": sp.label, "group": sp.group, "note": sp.test_acc_note})
        print(f"[Playground] Registered {sp.name} -> {sp.adapter_dir}", flush=True)

    # Optionally load Kev multimodal pointer-head models if KEV_ROOT is available
    global KEV_HELPERS
    if KEV_ROOT.exists() and (LEGACY_RUNS_DIR / "mm-4b-all-700-start").exists():
        try:
            if str(KEV_ROOT) not in sys.path:
                sys.path.insert(0, str(KEV_ROOT))
            from kev.api import SystemOneRequest as KevReq, output_tokens, to_answers, to_record
            from kev.checkpoint import LoadOptions
            from kev.model import SPECIAL, training_context
            from kev.predictors import LocalPredictor
            from kev.vision import image_inputs as kev_img_inputs

            ctx = training_context(1024)
            KEV_HELPERS = {
                "SystemOneRequest": KevReq, "output_tokens": output_tokens, "to_answers": to_answers,
                "to_record": to_record, "SPECIAL": SPECIAL, "CONTEXT": ctx, "image_inputs": kev_img_inputs,
            }
            for kname, kdir, klbl, knote in [
                ("kev-4b-mm", LEGACY_RUNS_DIR / "mm-4b-all-700-start", "对照架构 Kev-4B 多模态 (FIM + PointerHead)", "Kev 4B 对照 Test 85.33% / NLL 0.538"),
                ("kev-0.8b-mm", LEGACY_RUNS_DIR / "mm-all-700", "对照架构 Kev-0.8B 多模态 (FIM + PointerHead)", "Kev 0.8B 对照 Test 74.67%"),
            ]:
                if kdir.exists():
                    PREDICTORS[kname] = LocalPredictor(str(kdir), "cuda", LoadOptions(), ctx)
                    MODEL_CATALOG.append({"id": kname, "title": klbl, "group": "Kev Architecture", "note": knote})
                    print(f"[Playground] Registered Kev model {kname} -> {kdir}", flush=True)
        except Exception as exc:
            print(f"[Playground] Skipping optional Kev models: {exc}", flush=True)


def main():
    import uvicorn
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=23834)
    args = ap.parse_args()
    register_all_models()
    load_test_records()
    vram_gib = torch.cuda.memory_allocated() / (1024 ** 3)
    print(f"[Playground] Ready with {len(PREDICTORS)} methods ({vram_gib:.2f} GiB VRAM) on http://{args.host}:{args.port}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
