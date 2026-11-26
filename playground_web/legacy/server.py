#!/usr/bin/env python3
"""TypeSafe-style playground + System One compatible JSON API for the multimodal Kev checkpoints.

POST /v1/systemone   {"state": str | object, "questions": {...}, "model": "mm-all-700"}  ->  {"model", "answers", "usage", "evaluation_time_ms"}
POST /api/prompt     same body  ->  what the tokenizer produced: rendered text, special-token segments, per-token table
                     state may be a string or an object. An object may carry "image": "data:image/...;base64,..." (data URIs only).
GET  /v1/models      model names accepted by "model" ("kev-latest" is an alias of mm-all-700)
GET  /               playground page: State / Questions editors, Response / Prompt tabs, upload, model picker, Run request
GET  /api/sample/random  a random test-set sample (correct and wrong answers both come up)

Run (one idle GPU, all interfaces):
    CUDA_VISIBLE_DEVICES=7 /root/projects/kev/.venv/bin/python web/server.py --host 0.0.0.0 --port 23833

Scoring is the same path as kev.benchmark (encode -> forward -> to_answers). No authentication.
"""
import argparse, base64, io, json, random, threading, time
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from PIL import Image

from kev.api import SystemOneRequest, output_tokens, to_answers, to_record
from kev.checkpoint import LoadOptions
from kev.model import SPECIAL, training_context
from kev.predictors import LocalPredictor
from kev.vision import image_inputs
from nimble_backend import MM_ADAPTER, NIMBLE_MM_NAME, NimblePredictor

ROOT = Path(__file__).resolve().parents[1]          # /root/projects/jev-real-ad-1000
DATA = ROOT / "kev_data"
RUNS = ROOT / "runs"
MODELS = {"mm-all-700": RUNS / "mm-all-700", "mm-agreed-537": RUNS / "mm-agreed-537"}
DEFAULT_MODEL = "mm-all-700"
FIELDS = ["title", "body", "visual_ocr", "image_caption", "visual_risk_cues"]
CONTEXT = training_context(1024)                    # image tokens + state up to 1024 tokens
MIME = {"jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp", "gif": "image/gif", "bmp": "image/bmp"}
VISION_MARKERS = ["<|vision_start|>", "<|vision_end|>"]

PREDICTORS = {}
LOCK = threading.Lock()                             # one GPU: one request at a time
_SAMPLES = None

app = FastAPI(title="Kev playground (System One compatible)")


def image_data_uri(raw: bytes) -> str:
    try:
        with Image.open(io.BytesIO(raw)) as im:
            fmt = (im.format or "").lower()
            im.load()
    except Exception:
        raise HTTPException(400, "unreadable image")
    if fmt not in MIME:
        raise HTTPException(400, f"unsupported image format: {fmt or 'unknown'}")
    return f"data:{MIME[fmt]};base64," + base64.b64encode(raw).decode()


def resolve_model(model: str) -> str:
    if model == "kev-latest":
        return DEFAULT_MODEL
    if model in PREDICTORS:
        return model
    raise HTTPException(400, f"unknown model {model!r}; available: {list(PREDICTORS)}")


def encode_request(req: SystemOneRequest, name: str):
    """Validate, render and tokenize one request -> (predictor, rendered record, per-question meta, encoding)."""
    if isinstance(req.state, dict) and "image" in req.state:
        src = req.state["image"]
        if not isinstance(src, str) or not src.startswith("data:image/"):
            raise HTTPException(400, "state.image must be a data:image/...;base64 URI")
    p = PREDICTORS[name]
    try:
        rec, meta = to_record(req)
    except (ValueError, TypeError) as e:
        raise HTTPException(422, f"invalid request: {e}")
    with LOCK, torch.no_grad():
        try:
            enc = p.model.encode(p.tok, rec, max_state=CONTEXT["max_state"], max_branch=CONTEXT["max_branch"], strict=True)
        except ValueError as e:                     # ContextOverflow is a ValueError
            raise HTTPException(422, f"over context: {e}")
    if len(enc["ids"]) > CONTEXT["max_packed"]:
        raise HTTPException(422, f"packed request exceeds the {CONTEXT['max_packed']}-token limit")
    return p, rec, meta, enc


def answer(req: SystemOneRequest, name: str) -> dict:
    if isinstance(PREDICTORS[name], NimblePredictor):
        with LOCK, torch.no_grad():
            try:
                return PREDICTORS[name].answer(req, name)
            except ValueError as e:
                raise HTTPException(422, str(e))
    start = time.perf_counter()
    p, rec, meta, enc = encode_request(req, name)
    with LOCK, torch.no_grad():
        logits = p.model.forward(enc)
    probs = [torch.softmax(z.float(), -1).cpu().tolist() for z in logits]
    answers = to_answers(probs, meta)
    return {"model": name,
            "answers": answers,
            "usage": {"input_tokens": len(enc["ids"]), "output_tokens": output_tokens(p.tok, answers)},
            "evaluation_time_ms": round((time.perf_counter() - start) * 1000, 1)}


def opt_label(o: int) -> str:
    if o == -1:
        return ""
    if o == -2:
        return "decide"
    return f"option {o}"


def prompt_view(p, name: str, rec: dict, meta: list, enc: dict) -> dict:
    """What the tokenizer produced for this request, in three views: rendered text, special-token segments, token table."""
    tok = p.tok
    ids, seg, opt = enc["ids"], enc["seg"], enc["opt"]
    img_id = tok.convert_tokens_to_ids("<|image_pad|>")
    sp_ids = {tok.convert_tokens_to_ids(t) for t in SPECIAL + VISION_MARKERS}
    qids = [m["id"] for m in meta]

    def kind(t):
        return "image" if t == img_id else ("special" if t in sp_ids else "text")

    def seg_name(s):
        return "state" if s == 0 else f"question {qids[s - 1]}"

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
    image = None
    if rec.get("image"):
        image = {"image_tokens": n_img, "grid_patches_thw": list(image_inputs(rec["image"])[1])}
    rendered = {
        "state": rec["state"],
        "image": image,
        "questions": [{"id": m["id"], "type": m["type"], "instructions": q["instr"], "options": q["options"]}
                      for m, q in zip(meta, rec["questions"])],
    }
    return {
        "model": name,
        "summary": {"total_tokens": len(ids), "state_total": sum(1 for s in seg if s == 0),
                    "image_tokens": n_img, "questions": len(meta)},
        "rendered": rendered,
        "segments": groups,
        "tokens": rows,
    }


def resolve_and_encode(req: SystemOneRequest):
    name = resolve_model(req.model)
    return name, encode_request(req, name)


@app.post("/v1/systemone")
def systemone(req: SystemOneRequest):
    return answer(req, resolve_model(req.model))


@app.post("/api/prompt")
def prompt(req: SystemOneRequest):
    name = resolve_model(req.model)
    if isinstance(PREDICTORS[name], NimblePredictor):
        with LOCK, torch.no_grad():
            try:
                return PREDICTORS[name].prompt(req, name)
            except ValueError as e:
                raise HTTPException(422, str(e))
    name, (p, rec, meta, enc) = resolve_and_encode(req)
    return prompt_view(p, name, rec, meta, enc)


@app.get("/v1/models")
def models():
    return list(PREDICTORS)


_TEST = None


def load_test_records() -> list:
    """The jev test split (text fields, gold option, image file), loaded once."""
    global _TEST
    if _TEST is None:
        out = []
        with (DATA / "test_pos_start.jsonl").open(encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                state = dict(r["state"])
                image_rel = state.pop("image")
                q = next(iter(r["questions"].values()))
                out.append({"id": r["id"], "gold": q["label"], "state": state, "image_rel": image_rel,
                            "questions": {"industry": {"type": "choice", "instructions": q["instructions"], "criteria": q["criteria"]}}})
        _TEST = out
    return _TEST


@app.get("/api/sample/random")
def random_sample():
    """A uniformly random record of the test split: correct and wrong answers can both come up."""
    rec = random.choice(load_test_records())
    return {"id": rec["id"], "gold": rec["gold"], "state": rec["state"], "questions": rec["questions"],
            "image": image_data_uri((DATA / rec["image_rel"]).read_bytes())}


@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE


PAGE = r"""<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Kev Playground</title>
<style>
:root{--bd:#e4e4e7;--mute:#71717a;--bg:#fafafa;--fg:#18181b;--acc:#2563eb}
*{box-sizing:border-box}
html,body{height:100%;margin:0}
body{font-family:ui-sans-serif,-apple-system,system-ui,"PingFang SC","Microsoft YaHei",sans-serif;color:var(--fg);background:#fff;display:flex;flex-direction:column}
header{height:44px;display:flex;align-items:center;gap:12px;padding:0 16px;border-bottom:1px solid var(--bd);flex:0 0 auto}
header .title{font-weight:600;font-size:14px}
select,button{font:inherit}
select{padding:4px 8px;border:1px solid var(--bd);border-radius:6px;background:#fff;font-size:13px}
.btn{padding:6px 12px;border:1px solid var(--bd);background:#fff;border-radius:6px;cursor:pointer;font-size:13px;color:var(--fg)}
.btn:hover{background:var(--bg)}
.btn:disabled{opacity:.6;cursor:default}
.btn.pri{background:var(--acc);border-color:var(--acc);color:#fff;font-weight:500}
.btn.pri:hover{filter:brightness(1.05)}
.btn.sm{padding:3px 10px;font-size:12px}
.pill{padding:1px 8px;border:1px solid var(--bd);border-radius:999px;font-size:12px;color:var(--fg);background:#fff}
.pill.bad{color:#b91c1c;border-color:#fecaca;background:#fef2f2}
.pill.ok{color:#15803d;border-color:#bbf7d0;background:#f0fdf4}
.sp{flex:1}
main{flex:1;display:grid;grid-template-columns:1fr 1fr;min-height:0}
.col{display:flex;flex-direction:column;min-height:0}
.col.left{border-right:1px solid var(--bd)}
.panel{display:flex;flex-direction:column;min-height:0;flex:1}
.ph{display:flex;align-items:center;gap:8px;padding:8px 14px;border-bottom:1px solid var(--bd);font-size:12px;color:var(--mute);font-weight:500;text-transform:uppercase;letter-spacing:.04em;min-height:38px}
.ph .small{text-transform:none;letter-spacing:0;font-weight:400}
.tabs{display:flex;gap:4px;text-transform:none;letter-spacing:0}
.tabs button{border:1px solid transparent;background:none;padding:3px 10px;border-radius:6px;cursor:pointer;font-size:13px;color:var(--mute)}
.tabs button.on{background:#f4f4f5;border-color:var(--bd);color:var(--fg);font-weight:500}
.ed{position:relative;flex:1;display:flex;min-height:0;background:#fff}
.gut{width:44px;background:var(--bg);color:#a1a1aa;text-align:right;padding:10px 8px 10px 0;font:12.5px/1.6 ui-monospace,Menlo,monospace;overflow:hidden;user-select:none;white-space:pre;border-right:1px solid var(--bd)}
textarea.code{flex:1;border:0;outline:0;resize:none;padding:10px 12px;font:12.5px/1.6 ui-monospace,Menlo,monospace;white-space:pre;overflow:auto;background:transparent;tab-size:2}
.err{color:#b91c1c;font-size:12px;padding:4px 14px;min-height:20px}
#state-wrap{flex:0 0 36%;border-bottom:1px solid var(--bd)}
#q-wrap{flex:1}
.resp{flex:1;overflow:auto;margin:0;padding:10px 14px;font:12.5px/1.6 ui-monospace,Menlo,monospace;white-space:pre;background:#fff}
.resp .k{color:#7c3aed}.resp .s{color:#15803d}.resp .n{color:#b45309}.resp .b{color:#be185d}.resp .z{color:#be185d}
.ph-text{color:var(--mute);font-family:inherit;white-space:normal}
.status{font-size:12px;color:var(--mute);display:flex;gap:8px;align-items:center}
.pview{flex:1;overflow:auto;padding:10px 14px;font-size:13px;background:#fff}
.meta{font-size:12px;color:var(--mute);margin:4px 0 8px}
.sec{font-size:12px;font-weight:600;color:var(--mute);margin:16px 0 6px;text-transform:uppercase;letter-spacing:.03em}
.pt{background:#f4f4f5;border-radius:6px;padding:8px 10px;font:12.5px/1.6 ui-monospace,Menlo,monospace;white-space:pre-wrap;margin-bottom:8px;word-break:break-word}
.seg{margin:12px 0 4px;font-size:12px;color:#15803d;font-family:ui-monospace,Menlo,monospace}
.line{font:12.5px/1.9 ui-monospace,Menlo,monospace;white-space:pre-wrap;word-break:break-word;border:1px solid var(--bd);border-radius:6px;padding:8px 10px;background:#fff}
.tk-sp{background:#eef2ff;color:#4338ca;border:1px solid #c7d2fe;border-radius:4px;padding:0 4px;margin:0 1px;font-size:11.5px}
.tk-img{background:#fef3c7;color:#92400e;border:1px solid #fde68a;border-radius:4px;padding:0 4px;font-size:11.5px}
.tk-tx{color:var(--fg)}
details{margin-top:14px}
details summary{cursor:pointer;color:var(--mute);font-size:12.5px;margin-bottom:6px}
table.tbl{border-collapse:collapse;font:11.5px/1.5 ui-monospace,Menlo,monospace;width:100%}
table.tbl th,table.tbl td{border-bottom:1px solid var(--bd);padding:2px 6px;text-align:left;vertical-align:top}
table.tbl th{color:var(--mute);font-weight:500;position:sticky;top:0;background:#fff}
footer{height:56px;display:flex;align-items:center;gap:12px;padding:0 16px;border-top:1px solid var(--bd);flex:0 0 auto;background:#fff}
#chip{display:none;align-items:center;gap:6px;padding:4px 8px;border:1px solid var(--bd);border-radius:6px;font-size:12.5px;background:var(--bg)}
#chip img{width:28px;height:28px;object-fit:cover;border-radius:4px}
#chip button{border:0;background:none;cursor:pointer;color:var(--mute);font-size:14px;padding:0 2px}
.hint{font-size:12px;color:var(--mute)}
.judge{font-size:13px;padding:8px 14px;border-bottom:1px solid var(--bd)}
.judge.ok{color:#15803d;background:#f0fdf4}
.judge.bad{color:#b91c1c;background:#fef2f2}
@media (max-width:900px){main{grid-template-columns:1fr}.col.left{border-right:0;border-bottom:1px solid var(--bd)}}
</style></head>
<body>
<header>
  <span class="title">Kev Playground</span><span class="pill">多模态 · System One 兼容</span>
  <span class="sp"></span>
  <button class="btn" id="randsample">随机测试样本</button>
  <span class="hint" id="sample-stats"></span>
  <button class="btn" id="clear">Clear</button>
</header>
<main>
  <section class="col left">
    <div class="panel" id="state-wrap">
      <div class="ph">State <span class="sp"></span><span class="small" id="state-hint">object 或 string；上传的图片按底部“图片位置”插入（即 image 键在对象中的位置）</span></div>
      <div class="ed"><div class="gut" id="state-gut"></div><textarea class="code" id="state" spellcheck="false"></textarea></div>
      <div class="err" id="state-err"></div>
    </div>
    <div class="panel" id="q-wrap">
      <div class="ph">Questions <span class="sp"></span><button class="btn sm" id="addq">Add Question</button></div>
      <div class="ed"><div class="gut" id="q-gut"></div><textarea class="code" id="questions" spellcheck="false"></textarea></div>
      <div class="err" id="q-err"></div>
    </div>
  </section>
  <section class="col">
    <div class="ph">
      <span class="tabs">
        <button id="tab-resp" class="on">Response</button>
        <button id="tab-prompt">Prompt</button>
      </span>
      <span class="sp"></span>
      <span class="status" id="status"></span>
      <button class="btn sm" id="copy">Copy</button>
    </div>
    <div class="judge" id="judge" hidden></div>
    <pre class="resp" id="resp"><span class="ph-text">在左侧编辑 State 与 Questions，然后点击 Run request（⌘/Ctrl + Enter）。Prompt 标签页会显示 tokenizer 输出的带 special token 的原始 prompt。</span></pre>
    <div class="pview" id="prompt" hidden></div>
  </section>
</main>
<footer>
  <button class="btn" id="upl">Click to upload assets</button>
  <input type="file" id="file" accept="image/*" hidden>
  <div id="chip"><img id="thumb" alt=""><span id="chipname"></span><button id="rmimg" title="移除">×</button></div>
  <span class="hint" id="gold"></span>
  <span class="hint" id="image-hint" style="color:#b45309" hidden></span>
  <span class="sp"></span>
  <select id="imgpos" title="图片在 state 中的位置">
    <option value="start">图片：开头</option><option value="middle">图片：中间</option><option value="end">图片：结尾</option>
  </select>
  <select id="model" title="模型"></select>
  <button class="btn pri" id="run">Run request <span style="opacity:.7;font-size:11px">⌘↵</span></button>
</footer>
<script>
const FIELDS=["title","body","visual_ocr","image_caption","visual_risk_cues"];
const $=id=>document.getElementById(id);
const RE=/("(?:\\u[0-9a-fA-F]{4}|\\[^u]|[^\\"])*"(?:\s*:)?|\b(?:true|false|null)\b|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)/g;
let imageUri=null, samples=[], lastResp=null, lastPrompt=null, tab="resp";

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
  ta.addEventListener("keydown",e=>{ if(e.key==="Tab"){e.preventDefault();const s=ta.selectionStart;ta.value=ta.value.slice(0,s)+"  "+ta.value.slice(ta.selectionEnd);ta.selectionStart=ta.selectionEnd=s+2;upd()} });
  upd();
}
function setEditor(id,obj){
  $(id).value=JSON.stringify(obj,null,2);
  const isState=id==="state";
  gutter($(id),$(isState?"state-gut":"q-gut"));
  check($(id),$(isState?"state-err":"q-err"));
}

function setResponse(body,status,ms){
  const ok=status>=200&&status<300;
  $("status").textContent="";
  $("status").append(el("span",ok?"pill ok":"pill bad",`HTTP ${status}`));
  if(ms!=null)$("status").append(el("span","",`${ms.toFixed(0)} ms`));
  if(body&&body.evaluation_time_ms!=null)$("status").append(el("span","",`eval ${body.evaluation_time_ms} ms`));
  $("resp").innerHTML=hl(body);
  $("resp").dataset.raw=JSON.stringify(body,null,2);
}

let currentSample = null;
const stats = {ok: 0, bad: 0};
function judge(body, status){
  const box = $("judge");
  if(!currentSample || status < 200 || status >= 300 || !body.answers || !body.answers.industry){ box.hidden = true; return; }
  const a = body.answers.industry, pred = a.choice, gold = currentSample.gold, ok = pred === gold;
  const pg = a.probabilities && a.probabilities[gold];
  box.className = "judge " + (ok ? "ok" : "bad");
  box.textContent = (ok ? "✓ 正确" : "✗ 错误") + "：模型选 " + pred + (ok ? "" : "，真实答案 " + gold) + (pg != null ? "（真实答案的概率 " + (pg * 100).toFixed(1) + "%）" : "");
  box.hidden = false;
  if(!currentSample.counted){
    currentSample.counted = true;
    if(ok) stats.ok++; else stats.bad++;
    $("sample-stats").textContent = "本次随机：答对 " + stats.ok + " · 答错 " + stats.bad;
  }
}

function renderPrompt(v){
  const box=$("prompt"); box.textContent="";
  const s=v.summary;
  box.append(el("div","meta",`模型 ${v.model} · 总 token ${s.total_tokens}${s.state_total!=null?`（state ${s.state_total}，图片 ${s.image_tokens}）`:""}· 问题 ${s.questions} 个`));

  box.append(el("div","sec","1. 渲染后的文本（tokenizer 之前）"));
  box.append(el("div","pt",v.rendered.state));
  if(v.rendered.image) box.append(el("div","meta",`图片：${v.rendered.image.image_tokens} 个 image token，patch 网格 t×h×w = ${v.rendered.image.grid_patches_thw.join("×")}`));
  for(const q of v.rendered.questions){
    box.append(el("div","seg",`[${q.id}] type=${q.type} · ${q.instructions}`));
    box.append(el("div","pt",q.options.join("\n")));
  }

  box.append(el("div","sec","2. Tokenizer 输出（带 special token，按段落）"));
  let cur=null, line=null;
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

  const det=el("details"); det.append(el("summary","",`3. Token 明细（${v.tokens.length} 个）`));
  const tb=el("table","tbl");
  const head=el("tr"); for(const h of ["#","id","piece（原始）","段落","选项"]) head.append(el("th","",h));
  const thead=el("thead"); thead.append(head); tb.append(thead);
  const tbody=el("tbody");
  for(const r of v.tokens){
    const tr=el("tr");
    tr.append(el("td","",r.i)); tr.append(el("td","",r.id)); tr.append(el("td","",r.piece));
    tr.append(el("td","",r.segment)); tr.append(el("td","",r.opt));
    tbody.append(tr);
  }
  tb.append(tbody);
  det.append(tb); box.append(det);
}

function renderPromptTab(){
  if(!lastPrompt){ $("prompt").textContent="还没有运行过。点击 Run request。"; return }
  if(lastPrompt.status<200||lastPrompt.status>=300){
    $("prompt").textContent=""; $("prompt").append(el("div","pt",JSON.stringify(lastPrompt.body,null,2)));
    return;
  }
  renderPrompt(lastPrompt.body);
}

function showTab(t){
  tab=t;
  $("tab-resp").classList.toggle("on",t==="resp");
  $("tab-prompt").classList.toggle("on",t==="prompt");
  $("resp").hidden=t!=="resp";
  $("prompt").hidden=t!=="prompt";
  if(t==="prompt") renderPromptTab();
}

function setImage(uri,name){
  imageUri=uri; refreshHint();
  if(uri){$("thumb").src=uri;$("chipname").textContent=name||"image";$("chip").style.display="flex"}
  else{$("thumb").removeAttribute("src");$("chip").style.display="none";$("file").value=""}
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

async function run(){
  let state,questions;
  try{ state=JSON.parse($("state").value); }catch(e){ setResponse({error:"State 不是合法 JSON: "+e.message},0); return }
  try{ questions=JSON.parse($("questions").value); }catch(e){ setResponse({error:"Questions 不是合法 JSON: "+e.message},0); return }
  if(imageUri) state=withImage(state,imageUri,$("imgpos").value);
  const body=JSON.stringify({model:$("model").value,state,questions});
  const hdr={"Content-Type":"application/json"};
  $("run").disabled=true; $("status").textContent="running…";
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
    setResponse(ja,ra.status,ms);
    judge(ja,ra.status);
    if(tab==="prompt") renderPromptTab();
    $("prompt").dataset.ok=(rp.status>=200&&rp.status<300)?"1":"0";
  }catch(e){ setResponse({error:String(e)},0,performance.now()-t0) }
  finally{ $("run").disabled=false }
}

function addQuestion(){
  let q; try{ q=JSON.parse($("questions").value) }catch(e){ check($("questions"),$("q-err")); return }
  let i=1; while(q["new_noul_"+i]) i++;
  q["new_noul_"+i]={type:"noul",instructions:"这条广告是否涉及医疗功效宣传？",criteria:{"true":"","false":""}};
  setEditor("questions",q);
}

async function randomSample(){
  const s=await (await fetch("/api/sample/random")).json();
  currentSample=s; s.counted=false; $("judge").hidden=true;
  setEditor("state",s.state); setEditor("questions",s.questions);
  setImage(s.image,"sample-"+s.id.slice(-6)+".jpg");
  $("gold").textContent="测试集真实答案："+s.gold;
  $("status").textContent="";
  $("resp").innerHTML='<span class="ph-text">已载入样本，点击 Run request 运行。</span>';
  lastPrompt=null; if(tab==="prompt") renderPromptTab();
}

async function init(){
  bindEditor("state","state-err","state-gut");
  bindEditor("questions","q-err","q-gut");
  const models=await (await fetch("/v1/models")).json();
  $("model").innerHTML=models.map(m=>`<option value="${m}">${m}</option>`).join("");
  refreshHint();
  setEditor("state",{title:"正餐服务，开业五周年满减活动",body:"周末家庭聚餐套餐 8 折，到店享免费甜品。"});
  setEditor("questions",{industry:{type:"choice",instructions:"这条广告属于以下哪个行业类目？",criteria:{"正餐服务":"餐厅提供的堂食服务","家具零售":"家具销售","以上都不是":null}}});
  showTab("resp");
}

function needsImage(){ return $("model").value==="nimble-0.8b" && !imageUri; }
function refreshHint(){
  const hint=$("image-hint");
  hint.textContent=needsImage()?"未上传图片：nimble-0.8b 只看文本（训练时带图，准确率可能下降）":"";
  hint.hidden=!needsImage();
}
$("model").onchange=refreshHint;
$("run").onclick=run;
$("clear").onclick=()=>{ setImage(null); lastPrompt=null; $("resp").innerHTML='<span class="ph-text">已清空。</span>'; $("status").textContent=""; $("gold").textContent=""; if(tab==="prompt") renderPromptTab(); };
$("addq").onclick=addQuestion;
$("randsample").onclick=randomSample;
$("upl").onclick=()=>$("file").click();
$("file").onchange=()=>{ const f=$("file").files[0]; if(!f) return; const r=new FileReader(); r.onload=()=>setImage(r.result,f.name); r.readAsDataURL(f) };
$("rmimg").onclick=()=>setImage(null);
$("copy").onclick=()=>{ navigator.clipboard&&navigator.clipboard.writeText(tab==="prompt"?JSON.stringify(lastPrompt&&lastPrompt.body,null,2):($("resp").dataset.raw||$("resp").textContent)) };
$("tab-resp").onclick=()=>showTab("resp");
$("tab-prompt").onclick=()=>showTab("prompt");
document.addEventListener("keydown",e=>{ if((e.metaKey||e.ctrlKey)&&e.key==="Enter"){ e.preventDefault(); run() } });
init();
</script>
</body></html>
"""


def main():
    import uvicorn
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=23833)
    args = ap.parse_args()
    for name, path in MODELS.items():
        PREDICTORS[name] = LocalPredictor(str(path), "cuda", LoadOptions(), CONTEXT)
        print(f"loaded {name} from {path}", flush=True)
    PREDICTORS[NIMBLE_MM_NAME] = NimblePredictor(MM_ADAPTER, multimodal=True)
    print(f"loaded {NIMBLE_MM_NAME} (nimble with the advertisement image) from {MM_ADAPTER}", flush=True)
    load_test_records()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
