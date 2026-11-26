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
from fastapi.responses import HTMLResponse
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


@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE


PAGE = r"""<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Nimble 多模态 SOTA 交互体验台</title>
<style>
:root{--bd:#e4e4e7;--mute:#71717a;--bg:#fafafa;--fg:#18181b;--acc:#2563eb;--acc2:#7c3aed}
*{box-sizing:border-box}
html,body{height:100%;margin:0}
body{font-family:ui-sans-serif,-apple-system,system-ui,"PingFang SC","Microsoft YaHei",sans-serif;color:var(--fg);background:#fff;display:flex;flex-direction:column}
header{min-height:48px;display:flex;flex-wrap:wrap;align-items:center;gap:10px;padding:6px 16px;border-bottom:1px solid var(--bd);flex:0 0 auto;background:#fcfcfd}
header .title{font-weight:700;font-size:14.5px;color:#1e293b}
select,button,input{font:inherit}
select{padding:5px 8px;border:1px solid var(--bd);border-radius:6px;background:#fff;font-size:12.5px}
.btn{padding:5px 11px;border:1px solid var(--bd);background:#fff;border-radius:6px;cursor:pointer;font-size:12.5px;color:var(--fg);display:inline-flex;align-items:center;gap:5px}
.btn:hover{background:var(--bg)}
.btn:disabled{opacity:.55;cursor:default}
.btn.pri{background:var(--acc);border-color:var(--acc);color:#fff;font-weight:500}
.btn.pri:hover{filter:brightness(1.06)}
.btn.cmp{background:var(--acc2);border-color:var(--acc2);color:#fff;font-weight:500}
.btn.cmp:hover{filter:brightness(1.06)}
.btn.warn{border-color:#fed7aa;background:#fff7ed;color:#c2410c}
.btn.warn:hover{background:#ffedd5}
.btn.sm{padding:3px 9px;font-size:12px}
.pill{padding:2px 8px;border:1px solid var(--bd);border-radius:999px;font-size:11.5px;color:var(--fg);background:#fff}
.pill.bad{color:#b91c1c;border-color:#fecaca;background:#fef2f2}
.pill.ok{color:#15803d;border-color:#bbf7d0;background:#f0fdf4}
.pill.sota{color:#1d4ed8;border-color:#bfdbfe;background:#eff6ff;font-weight:500}
.sp{flex:1}
main{flex:1;display:grid;grid-template-columns:46% 54%;min-height:0}
.col{display:flex;flex-direction:column;min-height:0}
.col.left{border-right:1px solid var(--bd)}
.panel{display:flex;flex-direction:column;min-height:0;flex:1}
.ph{display:flex;align-items:center;gap:8px;padding:7px 14px;border-bottom:1px solid var(--bd);font-size:12px;color:var(--mute);font-weight:600;text-transform:uppercase;letter-spacing:.03em;min-height:38px;background:#fafafa}
.ph .small{text-transform:none;letter-spacing:0;font-weight:400}
.tabs{display:flex;gap:4px;text-transform:none;letter-spacing:0}
.tabs button{border:1px solid transparent;background:none;padding:4px 10px;border-radius:6px;cursor:pointer;font-size:12.5px;color:var(--mute)}
.tabs button.on{background:#fff;border-color:var(--bd);color:var(--fg);font-weight:600;box-shadow:0 1px 2px rgba(0,0,0,.04)}
.ed{position:relative;flex:1;display:flex;min-height:0;background:#fff}
.gut{width:40px;background:var(--bg);color:#a1a1aa;text-align:right;padding:10px 7px 10px 0;font:12px/1.6 ui-monospace,Menlo,monospace;overflow:hidden;user-select:none;white-space:pre;border-right:1px solid var(--bd)}
textarea.code{flex:1;border:0;outline:0;resize:none;padding:10px 12px;font:12px/1.6 ui-monospace,Menlo,monospace;white-space:pre;overflow:auto;background:transparent;tab-size:2}
.err{color:#b91c1c;font-size:12px;padding:2px 14px;min-height:18px}
#state-wrap{flex:0 0 42%;border-bottom:1px solid var(--bd)}
#q-wrap{flex:1}
.preview-strip{display:flex;gap:12px;align-items:center;padding:8px 14px;border-bottom:1px solid var(--bd);background:#f8fafc}
.preview-strip img{max-height:86px;max-width:140px;border-radius:6px;border:1px solid var(--bd);object-fit:contain;background:#fff;cursor:zoom-in}
.resp-wrap{flex:1;overflow:auto;display:flex;flex-direction:column}
.bars{padding:10px 14px;border-bottom:1px solid var(--bd);background:#fafafa}
.bar-row{display:grid;grid-template-columns:170px 1fr 56px;align-items:center;gap:8px;font-size:12px;margin:4px 0}
.bar-lbl{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-weight:500}
.bar-track{height:8px;background:#e4e4e7;border-radius:99px;overflow:hidden}
.bar-fill{height:100%;background:var(--acc);border-radius:99px}
.bar-fill.gold{background:#16a34a}
.bar-fill.wrong{background:#dc2626}
.resp{flex:1;margin:0;padding:10px 14px;font:12px/1.6 ui-monospace,Menlo,monospace;white-space:pre;background:#fff}
.resp .k{color:#7c3aed}.resp .s{color:#15803d}.resp .n{color:#b45309}.resp .b{color:#be185d}.resp .z{color:#be185d}
.ph-text{color:var(--mute);font-family:inherit;white-space:normal}
.status{font-size:12px;color:var(--mute);display:flex;gap:8px;align-items:center}
.pview,.cview{flex:1;overflow:auto;padding:12px 14px;font-size:13px;background:#fff}
.meta{font-size:12px;color:var(--mute);margin:4px 0 8px}
.sec{font-size:12px;font-weight:600;color:var(--mute);margin:14px 0 6px;text-transform:uppercase;letter-spacing:.03em}
.pt{background:#f4f4f5;border-radius:6px;padding:8px 10px;font:12px/1.6 ui-monospace,Menlo,monospace;white-space:pre-wrap;margin-bottom:8px;word-break:break-word}
.seg{margin:12px 0 4px;font-size:12px;color:#15803d;font-family:ui-monospace,Menlo,monospace}
.line{font:12px/1.9 ui-monospace,Menlo,monospace;white-space:pre-wrap;word-break:break-word;border:1px solid var(--bd);border-radius:6px;padding:8px 10px;background:#fff}
.tk-sp{background:#eef2ff;color:#4338ca;border:1px solid #c7d2fe;border-radius:4px;padding:0 4px;margin:0 1px;font-size:11px}
.tk-img{background:#fef3c7;color:#92400e;border:1px solid #fde68a;border-radius:4px;padding:0 4px;margin:0 1px;font-size:11px}
.tk-tx{color:var(--fg)}
table.tbl{border-collapse:collapse;font:12px/1.5 ui-sans-serif,system-ui,sans-serif;width:100%}
table.tbl th,table.tbl td{border-bottom:1px solid var(--bd);padding:7px 8px;text-align:left;vertical-align:middle}
table.tbl th{color:var(--mute);font-weight:600;position:sticky;top:0;background:#fafafa;font-size:11.5px}
table.tbl tr.row-sota{background:#f8faff}
footer{min-height:54px;display:flex;flex-wrap:wrap;align-items:center;gap:10px;padding:6px 16px;border-top:1px solid var(--bd);flex:0 0 auto;background:#fcfcfd}
.chk{display:inline-flex;align-items:center;gap:4px;font-size:12.5px;cursor:pointer;user-select:none;padding:3px 8px;border:1px solid var(--bd);border-radius:6px;background:#fff}
.hint{font-size:12px;color:var(--mute)}
.judge{font-size:13px;padding:8px 14px;border-bottom:1px solid var(--bd);font-weight:500}
.judge.ok{color:#15803d;background:#f0fdf4}
.judge.bad{color:#b91c1c;background:#fef2f2}
#modal{display:none;position:fixed;inset:0;background:rgba(0,0,0,.75);z-index:99;align-items:center;justify-content:center;cursor:zoom-out}
#modal img{max-width:92vw;max-height:92vh;border-radius:8px;background:#fff}
@media (max-width:960px){main{grid-template-columns:1fr}.col.left{border-right:0;border-bottom:1px solid var(--bd)}}
</style></head>
<body>
<header>
  <span class="title">Nimble 多模态 SOTA 交互体验台</span>
  <span class="pill sota">sandwich_v3 · FP32 · Vision-LoRA · 零泄漏锁温</span>
  <span class="sp"></span>
  <select id="sample-mode" title="测试集抽样模式">
    <option value="disagree">🎯 经典分歧样本（旧版答错 · SOTA答对）</option>
    <option value="poster_only">🖼️ 无标题正文纯海报样本</option>
    <option value="all">🎲 全部测试集随机样本 (150)</option>
  </select>
  <button class="btn" id="randsample">抽取测试样本</button>
  <button class="btn warn" id="strip-ocr" title="删除预提取的 ocr / caption，保留原始投放标题正文（模拟线上无 VLM 预处理场景）">✂️ 剥离 OCR/Caption</button>
  <button class="btn warn" id="strip-all" title="清空全部文本字段（state={}），令纯文本模型彻底盲猜，检验多模态模型纯靠像素识别海报的能力">🖼️ 纯看图（清空全部文字）</button>
  <button class="btn" id="restore-ocr" hidden>↺ 恢复完整字段</button>
  <span class="hint" id="sample-stats"></span>
</header>
<main>
  <section class="col left">
    <div class="preview-strip" id="img-strip" style="display:none">
      <img id="big-thumb" alt="广告原图" title="点击放大查看广告原图">
      <div style="flex:1;min-width:0">
        <div style="font-size:12.5px;font-weight:600" id="img-title">当前广告原图</div>
        <div class="hint" id="gold" style="margin-top:3px;color:#0f766e;font-weight:500"></div>
        <div style="margin-top:5px;display:flex;gap:6px">
          <button class="btn sm" id="upl2">更换图片</button>
          <button class="btn sm" id="rmimg">移除图片</button>
        </div>
      </div>
    </div>
    <div class="panel" id="state-wrap">
      <div class="ph">State (业务输入) <span class="sp"></span><button class="btn sm" id="upl">上传广告图</button><input type="file" id="file" accept="image/*" hidden></div>
      <div class="ed"><div class="gut" id="state-gut"></div><textarea class="code" id="state" spellcheck="false"></textarea></div>
      <div class="err" id="state-err"></div>
    </div>
    <div class="panel" id="q-wrap">
      <div class="ph">Questions (结构化决策定义：支持 choice / noul / score) <span class="sp"></span><button class="btn sm" id="addq">+ 新增布尔题</button></div>
      <div class="ed"><div class="gut" id="q-gut"></div><textarea class="code" id="questions" spellcheck="false"></textarea></div>
      <div class="err" id="q-err"></div>
    </div>
  </section>
  <section class="col">
    <div class="ph">
      <span class="tabs">
        <button id="tab-resp" class="on">单模型结果</button>
        <button id="tab-cmp">⚡ 全部方法同题横评</button>
        <button id="tab-prompt">Prompt & Token 透视</button>
      </span>
      <span class="sp"></span>
      <span class="status" id="status"></span>
      <button class="btn sm" id="copy">复制 JSON</button>
    </div>
    <div class="judge" id="judge" hidden></div>
    <div class="resp-wrap" id="resp-wrap">
      <div class="bars" id="top-bars" hidden></div>
      <pre class="resp" id="resp"><span class="ph-text">点击右上角「抽取测试样本」，然后点击右下角「单模型运行」或「⚡ 一键横评全部方法」直观对比旧版 v1、纯文本与多模态 SOTA 的真实表现！</span></pre>
    </div>
    <div class="cview" id="cmp-view" hidden>
      <div class="ph-text">点击底部紫色按钮「⚡ 一键横评全部方法」，即可在同一条广告上并行对比所有已加载模型的预测类别、校准置信度、Token 长度与延迟。</div>
    </div>
    <div class="pview" id="prompt" hidden></div>
  </section>
</main>
<footer>
  <label class="chk" title="启用在验证集上拟合锁定的零泄漏温度系数 T*，消除微调过自信（不改变 argmax 预测）">
    <input type="checkbox" id="chk-cal" checked> 零泄漏锁温校准 (T*)
  </label>
  <label class="chk" title="对 26 个候选项做 5 次均匀循环移位并融合去均值 Logits，消除字母位置偏置">
    <input type="checkbox" id="chk-tta"> 5 视角循环 TTA5
  </label>
  <select id="imgpos" title="图片在 state 对象中的键位置">
    <option value="start">图片：开头</option><option value="middle">图片：中间</option><option value="end">图片：结尾</option>
  </select>
  <span class="sp"></span>
  <select id="model" title="选择单个体验模型" style="max-width:340px;font-weight:500"></select>
  <button class="btn pri" id="run">单模型运行 ⌘↵</button>
  <button class="btn cmp" id="run-cmp">⚡ 一键横评全部方法</button>
</footer>
<div id="modal"><img id="modal-img" alt=""></div>
<script>
const $=id=>document.getElementById(id);
const RE=/("(?:\\u[0-9a-fA-F]{4}|\\[^u]|[^\\"])*"(?:\s*:)?|\b(?:true|false|null)\b|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)/g;
let imageUri=null, currentSample=null, savedFullState=null, lastResp=null, lastPrompt=null, catalog=[], tab="resp";
const stats={ok:0,bad:0};

function el(tag,cls,text){const e=document.createElement(tag);if(cls)e.className=cls;if(text!=null)e.textContent=String(text);return e}
function hl(obj){
  const s=JSON.stringify(obj,null,2).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
  return s.replace(RE,m=>{
    let c="n";
    if(m[0]==='"') c=/:$/.test(m)?"k":"s";
    else if(m==="true"||m==="false") c="b";
    else if(m==="null") c="z";
    return `<span class="${c}">${m}</span>`;
  });
}
function gutter(ta,g){
  const n=(ta.value.match(/\n/g)||[]).length+1;
  let t="";for(let i=1;i<=n;i++)t+=i+"\n";
  g.textContent=t; g.scrollTop=ta.scrollTop;
}
function check(ta,errEl){
  try{JSON.parse(ta.value);errEl.textContent="";return true}
  catch(e){errEl.textContent="JSON 格式错误："+e.message;return false}
}
function bindEditor(id,errId,gutId){
  const ta=$(id),g=$(gutId);
  const upd=()=>{gutter(ta,g);check(ta,$(errId))};
  ta.addEventListener("input",upd);
  ta.addEventListener("scroll",()=>{g.scrollTop=ta.scrollTop});
  ta.addEventListener("keydown",e=>{if(e.key==="Tab"){e.preventDefault();const s=ta.selectionStart;ta.value=ta.value.slice(0,s)+"  "+ta.value.slice(ta.selectionEnd);ta.selectionStart=ta.selectionEnd=s+2;upd()}});
  upd();
}
function setEditor(id,obj){
  $(id).value=JSON.stringify(obj,null,2);
  const isState=id==="state";
  gutter($(id),$(isState?"state-gut":"q-gut"));
  check($(id),$(isState?"state-err":"q-err"));
}
function setImage(uri,name){
  imageUri=uri;
  if(uri){
    $("big-thumb").src=uri; $("modal-img").src=uri;
    $("img-title").textContent="当前广告图："+(name||"uploaded");
    $("img-strip").style.display="flex";
  }else{
    $("big-thumb").removeAttribute("src");
    $("img-strip").style.display="none";
    $("file").value="";
  }
}
function withImage(state,uri,pos){
  if(state&&typeof state==="object"&&!Array.isArray(state)){
    const e=Object.entries(state).filter(([k])=>k!=="image");
    const at=pos==="start"?0:pos==="end"?e.length:Math.floor(e.length/2);
    e.splice(at,0,["image",uri]);
    return Object.fromEntries(e);
  }
  return {image:uri,state:state};
}
function renderTopBars(body){
  const box=$("top-bars"); box.innerHTML="";
  if(!body||!body.answers) { box.hidden=true; return; }
  const firstQ=Object.values(body.answers)[0];
  if(!firstQ||!firstQ.probabilities){ box.hidden=true; return; }
  const entries=Object.entries(firstQ.probabilities).sort((a,b)=>b[1]-a[1]).slice(0,5);
  const gold=currentSample&&currentSample.gold;
  const hdr=el("div","meta",`Top-5 候选概率分布${body.method_meta?`（Prompt: ${body.method_meta.prompt_format} · T=${body.method_meta.temperature_applied} · TTA=${body.method_meta.tta_views_used}）`:""}`);
  box.append(hdr);
  for(const [k,p] of entries){
    const row=el("div","bar-row");
    const lbl=el("div","bar-lbl",(gold&&k===gold?"★ ":"")+k);
    lbl.title=k;
    const tr=el("div","bar-track");
    const fill=el("div","bar-fill"+(gold?(k===gold?" gold":(k===firstQ.choice?" wrong":"")):""));
    fill.style.width=Math.max(1,p*100).toFixed(1)+"%";
    tr.append(fill);
    const pct=el("div","",(p*100).toFixed(1)+"%");
    pct.style.textAlign="right";
    row.append(lbl,tr,pct);
    box.append(row);
  }
  box.hidden=false;
}
function setResponse(body,status,ms){
  const ok=status>=200&&status<300;
  $("status").textContent="";
  $("status").append(el("span",ok?"pill ok":"pill bad",`HTTP ${status}`));
  if(body&&body.usage)$("status").append(el("span","pill",`${body.usage.input_tokens} tok`));
  if(body&&body.evaluation_time_ms!=null)$("status").append(el("span","",`GPU ${body.evaluation_time_ms} ms`));
  else if(ms!=null)$("status").append(el("span","",`${ms.toFixed(0)} ms`));
  renderTopBars(ok?body:null);
  $("resp").innerHTML=hl(body);
  $("resp").dataset.raw=JSON.stringify(body,null,2);
}
function judge(body,status){
  const box=$("judge");
  if(!currentSample||status<200||status>=300||!body.answers||!body.answers.industry){box.hidden=true;return}
  const a=body.answers.industry, pred=a.choice, gold=currentSample.gold, ok=pred===gold;
  const pg=a.probabilities&&a.probabilities[gold];
  box.className="judge "+(ok?"ok":"bad");
  box.textContent=(ok?"✓ 预测正确":"✗ 预测错误")+"：模型选择「"+pred+"」"+(ok?"":" ｜ 测试集真实标签：「"+gold+"」")+(pg!=null?"（赋予真实标签概率 "+(pg*100).toFixed(1)+"%）":"");
  box.hidden=false;
}
function renderPrompt(v){
  const box=$("prompt"); box.textContent="";
  const s=v.summary;
  box.append(el("div","meta",`模型 ${v.model} · 格式 ${s.prompt_format||""} · 总 Token ${s.total_tokens}（其中视觉 Token ${s.image_tokens}）· FP32候选头=${s.fp32_head}`));
  box.append(el("div","sec","1. 结构化分段 Token 流（直观对比 sandwich_v3 与旧版 v1）"));
  let cur=null,line=null;
  for(const g of v.segments){
    if(g.segment!==cur){
      cur=g.segment;
      box.append(el("div","seg","── "+cur));
      line=el("div","line"); box.append(line);
    }
    if(g.kind==="special") line.append(el("span","tk-sp",g.text));
    else if(g.kind==="image") line.append(el("span","tk-img",`${g.text} ×${g.count}`));
    else line.append(el("span","tk-tx",g.text));
  }
  const det=el("details"); det.append(el("summary","",`2. 展开逐 Token 词表映射明细（共 ${v.tokens.length} 个）`));
  const tb=el("table","tbl");
  const head=el("tr"); for(const h of ["#","Token ID","Piece","段落"]) head.append(el("th","",h));
  const thead=el("thead"); thead.append(head); tb.append(thead);
  const tbody=el("tbody");
  for(const r of v.tokens){
    const tr=el("tr");
    tr.append(el("td","",r.i),el("td","",r.id),el("td","",r.piece),el("td","",r.segment));
    tbody.append(tr);
  }
  tb.append(tbody); det.append(tb); box.append(det);
}
function showTab(t){
  tab=t;
  $("tab-resp").classList.toggle("on",t==="resp");
  $("tab-cmp").classList.toggle("on",t==="cmp");
  $("tab-prompt").classList.toggle("on",t==="prompt");
  $("resp-wrap").hidden=t!=="resp";
  $("cmp-view").hidden=t!=="cmp";
  $("prompt").hidden=t!=="prompt";
  if(t==="prompt"){
    if(!lastPrompt) $("prompt").textContent="请先点击「单模型运行」以生成当前所选模型的 Prompt 透视图。";
    else if(lastPrompt.status>=200&&lastPrompt.status<300) renderPrompt(lastPrompt.body);
  }
}
function parsePayload(){
  let state,questions;
  try{state=JSON.parse($("state").value)}catch(e){alert("State JSON 错误: "+e.message);return null}
  try{questions=JSON.parse($("questions").value)}catch(e){alert("Questions JSON 错误: "+e.message);return null}
  if(imageUri) state=withImage(state,imageUri,$("imgpos").value);
  return {
    state, questions,
    calibrated: $("chk-cal").checked,
    tta_views: $("chk-tta").checked ? 5 : 1
  };
}
async function runSingle(){
  const p=parsePayload(); if(!p) return;
  const body=JSON.stringify({...p, model:$("model").value});
  const hdr={"Content-Type":"application/json"};
  $("run").disabled=true; $("status").textContent="推理中…";
  const t0=performance.now();
  try{
    const [ra,rp]=await Promise.all([
      fetch("/v1/systemone",{method:"POST",headers:hdr,body}),
      fetch("/api/prompt",{method:"POST",headers:hdr,body})
    ]);
    const ms=performance.now()-t0;
    const ja=await ra.json().catch(()=>({detail:ra.statusText}));
    const jp=await rp.json().catch(()=>({detail:rp.statusText}));
    lastResp={body:ja,status:ra.status};
    lastPrompt={body:jp,status:rp.status};
    if(tab==="cmp") showTab("resp");
    setResponse(ja,ra.status,ms);
    judge(ja,ra.status);
    if(tab==="prompt"&&jp) renderPrompt(jp);
  }catch(e){setResponse({error:String(e)},0,performance.now()-t0)}
  finally{$("run").disabled=false}
}
async function runCompareAll(){
  const p=parsePayload(); if(!p) return;
  showTab("cmp");
  $("judge").hidden=true;
  $("run-cmp").disabled=true;
  $("cmp-view").innerHTML='<div class="ph-text">⚡ 正在同一条样本上并行评测全部方法，请稍候（约 1~3 秒）…</div>';
  $("status").textContent="全方法横评中…";
  try{
    const r=await fetch("/api/compare",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(p)});
    const data=await r.json();
    $("status").innerHTML=`<span class="pill ok">横评完成</span><span>总耗时 ${data.total_wall_ms} ms</span>`;
    const box=$("cmp-view"); box.innerHTML="";
    const qid=Object.keys(p.questions)[0];
    const gold=currentSample&&currentSample.gold;
    const info=el("div","meta",`评估问题：${qid}${gold?` ｜ 真实答案：「${gold}」`:""} ｜ 零泄漏锁温=${data.calibrated?"开启":"关闭"} ｜ 视角数=${data.tta_views}`);
    box.append(info);
    const tb=el("table","tbl");
    const hr=el("tr");
    for(const h of ["方法 / 模型","格式 & 视觉策略","预测结果","Top-1 概率","真实标签概率","输入 Token","GPU 延迟","操作"]) hr.append(el("th","",h));
    const thd=el("thead"); thd.append(hr); tb.append(thd);
    const tbd=el("tbody");
    for(const item of data.results){
      const tr=el("tr",item.model.includes("sota")?"row-sota":"");
      const catInfo=catalog.find(x=>x.id===item.model)||{};
      const nameTd=el("td","");
      nameTd.innerHTML=`<div style="font-weight:600">${catInfo.title||item.model}</div><div style="font-size:11px;color:#71717a">${catInfo.note||""}</div>`;
      if(!item.ok){
        tr.append(nameTd,el("td","","—"),el("td","",`错误: ${item.error}`),el("td","","—"),el("td","","—"),el("td","","—"),el("td","","—"),el("td","",""));
        tbd.append(tr); continue;
      }
      const res=item.result, ans=res.answers[qid]||{}, meta=res.method_meta||{};
      const pred=ans.type==="choice"?ans.choice:(ans.type==="noul"?`P(true)=${ans.noul}`:`score=${ans.score}`);
      const probs=ans.probabilities||{};
      const topP=ans.type==="choice"&&probs[ans.choice]!=null?(probs[ans.choice]*100).toFixed(1)+"%":"—";
      const goldP=(gold&&probs[gold]!=null)?(probs[gold]*100).toFixed(1)+"%":"—";
      const predTd=el("td","");
      if(gold&&ans.type==="choice"){
        const isOk=pred===gold;
        predTd.innerHTML=`<span class="pill ${isOk?'ok':'bad'}">${isOk?'✓':'✗'} ${pred}</span>`;
      }else{
        predTd.textContent=pred;
      }
      const fmtTd=el("td","",`${meta.prompt_format||""}${meta.multimodal?" · 图文":" · 纯文本"} (T=${meta.temperature_applied})`);
      const actTd=el("td","");
      const btnP=el("button","btn sm","看 Prompt");
      btnP.onclick=()=>{ $("model").value=item.model; runSingle().then(()=>showTab("prompt")); };
      actTd.append(btnP);
      tr.append(nameTd,fmtTd,predTd,el("td","",topP),el("td","",goldP),el("td","",`${res.usage.input_tokens}`),el("td","",`${res.evaluation_time_ms} ms`),actTd);
      tbd.append(tr);
    }
    tb.append(tbd); box.append(tb);
  }catch(e){
    $("cmp-view").textContent="横评请求失败: "+e;
  }finally{
    $("run-cmp").disabled=false;
  }
}
async function randomSample(){
  const mode=$("sample-mode").value;
  const s=await (await fetch(`/api/sample/random?mode=${encodeURIComponent(mode)}`)).json();
  currentSample=s; savedFullState=JSON.parse(JSON.stringify(s.state));
  $("restore-ocr").hidden=true; $("strip-ocr").hidden=false; $("strip-all").hidden=false;
  $("judge").hidden=true; $("top-bars").hidden=true;
  setEditor("state",s.state); setEditor("questions",s.questions);
  setImage(s.image,`${s.id}.jpg`);
  $("gold").textContent=`测试集真实标签：${s.gold}${s.is_disagree?"（🔥 旧版 v1 答错 · SOTA 逆转样本）":""}${!s.has_copy?"（🖼️ 无标题正文纯海报）":""}`;
  $("status").textContent="";
  $("resp").innerHTML='<span class="ph-text">样本已载入！点击右下角「⚡ 一键横评全部方法」直接对比各模型表现。</span>';
  lastPrompt=null;
}
function stripOcr(){
  try{
    const cur=JSON.parse($("state").value);
    if(!savedFullState) savedFullState=JSON.parse(JSON.stringify(cur));
    const stripped={};
    for(const k of ["title","body"]) if(cur[k]) stripped[k]=cur[k];
    setEditor("state",stripped);
    $("restore-ocr").hidden=false;
  }catch(e){}
}
function stripAllText(){
  try{
    const cur=JSON.parse($("state").value);
    if(!savedFullState) savedFullState=JSON.parse(JSON.stringify(cur));
    setEditor("state",{});
    $("restore-ocr").hidden=false;
  }catch(e){}
}
function restoreOcr(){
  if(savedFullState){ setEditor("state",savedFullState); $("restore-ocr").hidden=true; }
}
function addQuestion(){
  let q; try{q=JSON.parse($("questions").value)}catch(e){return}
  let i=1; while(q["risk_check_"+i]) i++;
  q["risk_check_"+i]={type:"noul",instructions:"该广告画面或文案中是否包含夸大绝对化用语或功效承诺风险？",criteria:{"true":"包含风险暗示或功效承诺","false":"正常合规展示"}};
  setEditor("questions",q);
}
async function init(){
  bindEditor("state","state-err","state-gut");
  bindEditor("questions","q-err","q-gut");
  const cat=await (await fetch("/api/catalog")).json();
  catalog=cat.models||[];
  $("model").innerHTML=catalog.map(m=>`<option value="${m.id}">${m.title}</option>`).join("");
  if(cat.sample_counts){
    $("sample-stats").textContent=`测试库 ${cat.sample_counts.all} 条（分歧样本 ${cat.sample_counts.disagree} 条 · 纯海报 ${cat.sample_counts.poster_only} 条）`;
  }
  await randomSample();
}
$("run").onclick=runSingle;
$("run-cmp").onclick=runCompareAll;
$("randsample").onclick=randomSample;
$("strip-ocr").onclick=stripOcr;
$("strip-all").onclick=stripAllText;
$("restore-ocr").onclick=restoreOcr;
$("addq").onclick=addQuestion;
$("upl").onclick=()=>$("file").click();
$("upl2").onclick=()=>$("file").click();
$("file").onchange=()=>{const f=$("file").files[0];if(!f)return;const r=new FileReader();r.onload=()=>{currentSample=null;$("gold").textContent="自定义上传图片";setImage(r.result,f.name)};r.readAsDataURL(f)};
$("rmimg").onclick=()=>setImage(null);
$("big-thumb").onclick=()=>{$("modal").style.display="flex"};
$("modal").onclick=()=>{$("modal").style.display="none"};
$("copy").onclick=()=>{navigator.clipboard&&navigator.clipboard.writeText(tab==="prompt"?JSON.stringify(lastPrompt&&lastPrompt.body,null,2):($("resp").dataset.raw||$("resp").textContent))};
$("tab-resp").onclick=()=>showTab("resp");
$("tab-cmp").onclick=()=>showTab("cmp");
$("tab-prompt").onclick=()=>showTab("prompt");
document.addEventListener("keydown",e=>{if((e.metaKey||e.ctrlKey)&&e.key==="Enter"){e.preventDefault();runSingle()}});
init();
</script>
</body></html>
"""


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
