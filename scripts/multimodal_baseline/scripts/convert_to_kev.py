"""Convert jev-real-ad-1000 (parquet) into Kev's labelled-request JSONL.

- State is text (title, body, visual_ocr, image_caption, visual_risk_cues); empty fields are dropped. The *_mm.jsonl files
  add the ad image: state.image = "images/<id>.<ext>" (relative to kev_data/, where the image is decoded from the parquet's
  image struct: kev_data/images/<id>.<ext>, the ext from the image's real format). The text-only files are unchanged.
- Leak-prone fields are never read: ground_truth_*, underlying_gt_*, deepseek_*, jev_*, ooc_reason,
  risk_tier (of the gold class), candidates_retrieval_scores.
- One `choice` question over the 25 retrieved candidates plus an explicit "none of the above" option.
  OOC is taken from `is_ooc` (the dataset uses both label_idx=-1 and label_idx=25 for OOC).
- train_all.jsonl = all 700 train rows; train_agreed.jsonl = dual_model_agreed==True only (537).
"""
import io, json, collections
from pathlib import Path
import pyarrow.parquet as pq
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OOC = "以上都不是"
INSTRUCTIONS = "这条广告属于以下哪个行业类目？"
DESC_CHARS = 40
COLS = ["id", "title", "body", "visual_ocr", "image_caption", "visual_risk_cues", "candidates",
        "label_idx", "ground_truth_cat_id", "is_ooc", "dual_model_agreed"]
IMAGE_DIR = ROOT / "kev_data/images"
IMAGE_EXT = {"JPEG": "jpg", "PNG": "png", "WEBP": "webp"}

taxonomy = {t["cat_id"]: t for t in json.load(open(ROOT / "taxonomy/ad_taxonomy_630.json", encoding="utf-8"))}


def short_desc(cat_id):
    d = (taxonomy.get(cat_id, {}).get("description") or "").replace("\n", " ").strip()
    return d[:DESC_CHARS] if d else None


def convert(r):
    state = {k: r[k].strip() for k in ["title", "body", "visual_ocr", "image_caption", "visual_risk_cues"]
             if r[k] and r[k].strip()}
    cats, names = r["candidates"]["cat_id"], r["candidates"]["name"]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate candidate names in {r['id']}")
    criteria = {n: short_desc(c) for c, n in zip(cats, names)}
    if OOC in criteria:
        raise ValueError(f"candidate collides with OOC option in {r['id']}")
    criteria[OOC] = "广告所属行业不在以上候选类目中"
    if r["is_ooc"]:
        label = OOC
    else:
        idx = r["label_idx"]
        if not (0 <= idx < len(cats)) or cats[idx] != r["ground_truth_cat_id"]:
            raise ValueError(f"label_idx/ground truth mismatch in {r['id']}: idx={idx} gt={r['ground_truth_cat_id']}")
        label = names[idx]
    return {"id": r["id"], "state": state,
            "questions": {"industry": {"type": "choice", "instructions": INSTRUCTIONS,
                                       "criteria": criteria, "label": label}}}


def read(split, columns=COLS):
    return pq.read_table(ROOT / f"data/{split}-00000-of-00001.parquet", columns=columns).to_pylist()


def write_image(r):
    """Decode the row's image struct (bytes, path) to kev_data/images/<id>.<ext> and return its path relative to kev_data.
    The format is the image's own (PIL), not the parquet's path suffix (the suffix is .jpg for PNG and WebP content too)."""
    data = r["image"]["bytes"]
    with Image.open(io.BytesIO(data)) as img:
        if img.format not in IMAGE_EXT: raise ValueError(f"unsupported image format {img.format} in {r['id']}")
        ext = IMAGE_EXT[img.format]
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    name = f"{r['id']}.{ext}"
    (IMAGE_DIR / name).write_bytes(data)
    return f"images/{name}"


def convert_mm(r):
    """convert(r) plus the row's image (state.image), for the *_mm.jsonl files."""
    rec = convert(r)
    rec["state"]["image"] = write_image(r)
    return rec


def write(path, recs):
    with path.open("w", encoding="utf-8") as f:
        for x in recs:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    labels = collections.Counter(x["questions"]["industry"]["label"] for x in recs)
    print(path.name, "n=", len(recs), "ooc=", labels[OOC], "distinct_labels=", len(labels),
          "max_share=", round(max(labels.values()) / len(recs), 3))


train = read("train")
val = read("validation")
test = read("test")
write(ROOT / "kev_data/train_all.jsonl", [convert(r) for r in train])
write(ROOT / "kev_data/train_agreed.jsonl", [convert(r) for r in train if r["dual_model_agreed"]])
write(ROOT / "kev_data/validation.jsonl", [convert(r) for r in val])
write(ROOT / "kev_data/test.jsonl", [convert(r) for r in test])

# multimodal: the same records with their image (decoded once per row; the text fields are unchanged)
train_img = read("train", COLS + ["image"]); val_img = read("validation", COLS + ["image"]); test_img = read("test", COLS + ["image"])
write(ROOT / "kev_data/train_all_mm.jsonl", [convert_mm(r) for r in train_img])
write(ROOT / "kev_data/train_agreed_mm.jsonl", [convert_mm(r) for r in train_img if r["dual_model_agreed"]])
write(ROOT / "kev_data/validation_mm.jsonl", [convert_mm(r) for r in val_img])
write(ROOT / "kev_data/test_mm.jsonl", [convert_mm(r) for r in test_img])
