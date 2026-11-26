#!/usr/bin/env python3
"""Probe 1 for the text-as-image idea: can Qwen3.5 read state text rendered as an image, and what does it cost?

For sampled test records, the state text (title, body, visual_ocr, image_caption, visual_risk_cues) is rendered with a CJK
font at several font sizes. The image goes through kev's own vision preprocessing (patch 16, merge 2, 32-aligned,
token cap), then the base model continues a "transcribe the text" prompt. Reported per font size:
  - character error rate (CER) of the transcription against the rendered text (whitespace removed),
  - visual tokens per character,
  - the rendered image size.

    CUDA_VISIBLE_DEVICES=1 /root/projects/kev/.venv/bin/python probe_text_image/probe.py --model 0.8b --n 12 --sizes 16 20 24
"""
import argparse, json, random, sys, time
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]                  # jev-real-ad-1000
sys.path.insert(0, "/root/projects/kev")
sys.path.insert(0, "/root/projects/nimble")
sys.path.insert(0, str(ROOT / "scripts"))

import kev.vision as V
from kev.vision import image_inputs, image_tokens
import nimble.training.schema_train as st
import nimble_mm as M                                        # for the base checkpoint revision used everywhere

FONT = "/root/.trae-server/bin/stable-11561ecda36d39c2469f5df9b26bc8ecec59006a-debian10/extensions/ai-completion/resource/aiserver/resources/font/HeiTi.ttf"
OUT = ROOT / "probe_text_image"
LABELS = [("title", "标题"), ("body", "正文"), ("visual_ocr", "图中文字"), ("image_caption", "画面描述"), ("visual_risk_cues", "风险线索")]
PROMPT = "\n以下是图片中的全部文字，逐字转写：\n"
WIDTH = 640


def state_text(state):
    return "\n".join(f"{label}：{state[key]}" for key, label in LABELS if state.get(key))


def render(text, px, path):
    font = ImageFont.truetype(FONT, px)
    margin, line_h = px // 2, int(px * 1.2)
    lines, cur = [], ""
    for para in text.split("\n"):
        cur = ""
        for ch in para:
            if cur and font.getlength(cur + ch) > WIDTH - 2 * margin:
                lines.append(cur)
                cur = ch
            else:
                cur += ch
        lines.append(cur)
    img = Image.new("RGB", (WIDTH, margin * 2 + line_h * len(lines)), "white")
    draw = ImageDraw.Draw(img)
    for i, line in enumerate(lines):
        draw.text((margin, margin + i * line_h), line, font=font, fill="black")
    img.save(path)
    return img.size


def levenshtein(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="0.8b", choices=["0.8b", "4b"])
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--sizes", type=int, nargs="+", default=[16, 20, 24])
    ap.add_argument("--token-cap", type=int, default=256, help="kev's image budget (IMAGE_TOKENS) for the probe")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    V.IMAGE_MAX_PIXELS = a.token_cap * (V.PATCH * V.MERGE) ** 2
    base = "Qwen/Qwen3.5-0.8B-Base" if a.model == "0.8b" else "Qwen/Qwen3.5-4B-Base"
    from transformers import AutoTokenizer
    snap = Path.home() / ".cache/huggingface/hub" / f"models--{base.replace('/', '--')}" / "snapshots"
    rev = M_REV if a.model == "0.8b" else sorted(p.name for p in snap.iterdir())[-1]   # offline: the cached snapshot
    tok = AutoTokenizer.from_pretrained(base, revision=rev)
    model = st.load_base(base, rev).eval()
    pad = tok.convert_tokens_to_ids("<|image_pad|>")

    font_cache = {px: ImageFont.truetype(FONT, px) for px in a.sizes}
    records = [json.loads(l) for l in open(ROOT / "kev_data/test_pos_start.jsonl", encoding="utf-8")]
    picked = random.Random(a.seed).sample(records, a.n)
    rows = []
    for px in a.sizes:
        font = font_cache[px]
        for rec in picked:
            raw = state_text(rec["state"])
            # keep only characters the font has a glyph for (emoji and symbols would render as boxes)
            text_kept = "".join(ch if (ch.isspace() or font.getmask(ch).getbbox() is not None) else "" for ch in raw)
            ref = "".join(ch for ch in text_kept if not ch.isspace())
            path = OUT / f"render_{a.model}_{px}px_{rec['id']}.png"
            size = render(text_kept, px, path)
            pixels, grid = image_inputs(str(path))
            n_img = image_tokens(grid)
            head, tail = tok.encode("<|vision_start|>", add_special_tokens=False), tok.encode("<|vision_end|>", add_special_tokens=False) + tok.encode(PROMPT, add_special_tokens=False)
            ids = head + [pad] * n_img + tail
            mm = [0] * len(head) + [1] * n_img + [0] * len(tail)
            max_new = int(len(ref) * 1.25) + 24
            t0 = time.time()
            with torch.no_grad():
                out = model.generate(input_ids=torch.tensor([ids], device="cuda"), attention_mask=torch.ones(1, len(ids), dtype=torch.long, device="cuda"),
                                     pixel_values=pixels.to("cuda", torch.bfloat16), image_grid_thw=torch.tensor([list(grid)], device="cuda"),
                                     mm_token_type_ids=torch.tensor([mm], device="cuda"), max_new_tokens=max_new, do_sample=False,
                                     pad_token_id=tok.pad_token_id or tok.eos_token_id)
            hyp_raw = tok.decode(out[0, len(ids):], skip_special_tokens=True)
            hyp = "".join(ch for ch in hyp_raw if not ch.isspace())[: int(len(ref) * 1.25) + 4]
            cer = levenshtein(ref, hyp) / max(1, len(ref))
            rows.append({"px": px, "id": rec["id"], "chars": len(ref), "image_tokens": n_img, "tokens_per_char": n_img / max(1, len(ref)),
                         "size": size, "cer": cer, "seconds": round(time.time() - t0, 1), "ref_head": ref[:40], "hyp_head": hyp[:40]})
            print(f"{px}px {rec['id']} chars={len(ref)} tokens={n_img} tok/char={n_img/max(1,len(ref)):.3f} size={size} CER={cer:.3f} {time.time()-t0:.1f}s", flush=True)
            print(f"   ref: {ref[:60]}\n   hyp: {hyp[:60]}", flush=True)

    summary = {}
    for px in a.sizes:
        r = [x for x in rows if x["px"] == px]
        summary[px] = {"n": len(r), "mean_cer": sum(x["cer"] for x in r) / len(r), "median_cer": sorted(x["cer"] for x in r)[len(r) // 2],
                       "mean_chars": sum(x["chars"] for x in r) / len(r), "mean_tokens": sum(x["image_tokens"] for x in r) / len(r),
                       "tokens_per_char": sum(x["image_tokens"] for x in r) / sum(x["chars"] for x in r)}
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    (OUT / f"results_{a.model}_cap{a.token_cap}.json").write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=1), encoding="utf-8")


M_REV = "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"

if __name__ == "__main__":
    main()
