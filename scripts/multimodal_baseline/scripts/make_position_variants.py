"""Image-position variants of the multimodal Kev data (kev_data/*_mm.jsonl).

The image key is moved among the state's text fields; the fields keep their order:
  start  -> image first (the layout every earlier multimodal checkpoint was trained on)
  middle -> image after half of the text fields
  end    -> image last
Writes kev_data/<split>_pos_<pos>.jsonl for train_all, validation and test.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KD = ROOT / "kev_data"


def place(state: dict, pos: str) -> dict:
    items = [(k, v) for k, v in state.items() if k != "image"]
    at = {"start": 0, "middle": len(items) // 2, "end": len(items)}[pos]
    items.insert(at, ("image", state["image"]))
    return dict(items)


for split in ["train_all", "validation", "test"]:
    recs = [json.loads(line) for line in (KD / f"{split}_mm.jsonl").open(encoding="utf-8")]
    for pos in ["start", "middle", "end"]:
        out = KD / f"{split}_pos_{pos}.jsonl"
        with out.open("w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps({**r, "state": place(r["state"], pos)}, ensure_ascii=False) + "\n")
        print(out.name, len(recs))
