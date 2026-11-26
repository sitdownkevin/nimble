"""Multimodal nimble: the advertisement image enters nimble's prompt as Qwen vision tokens.

Everything except the image is nimble's own: system prompt, JSON context and field schema, letter codes, candidate-logit
loss and evaluation (nimble.training.schema_train). This module only adds: the image block in the user message, the
expansion of its placeholder into one <|image_pad|> per merged patch, the pixel tensors, the mm_token_type_ids the model
needs to place image tokens on the 3-D rope, and a collator that carries them. Training and the playground both call
encode_one, so the prompt a model was trained on is the prompt it is served with.
"""
import random, string, sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
NIMBLE = Path("/root/projects/nimble")
KEV = Path("/root/projects/kev")
for p in (NIMBLE, KEV):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from nimble.evaluation.evaluate_pilot import adapt_input
from nimble.scoring.parallel_schema import SYSTEM_PROMPT, choice_key, choices_for, safe_json
from kev.vision import image_inputs, image_tokens

IMAGE_BLOCK = "<|vision_start|><|image_pad|><|vision_end|>"
IMAGE_PAD = "<|image_pad|>"
MARKER = "__PARALLEL_FIELD_TARGET__"
MAX_LENGTH = 2048          # nimble's training prompt limit (text + image tokens)
MAX_CHOICES = 26           # one letter code per option


def image_for(source):
    """-> (pixel values in the model's bf16, grid (t, h, w) in patches) for a data URI or an image file (kev's preprocessing)."""
    pixels, grid = image_inputs(source)
    return pixels.to(torch.bfloat16), tuple(grid)


def encode_one(tok, context, schema, image, shuffle_seed=None, row_id=""):
    """nimble's prepare_prompts for one field, plus the image block. `schema` is nimble's (one field, enum or boolean);
    `image` is image_for(...). Returns the prompt's token ids, the candidate token ids (one per option, in choice order),
    the choice values, and the multimodal rows."""
    names = list(schema)
    if len(names) != 1:
        raise ValueError("one decision question per prompt")
    name = names[0]
    field = schema[name]
    if shuffle_seed is not None:                       # nimble's per-record option shuffle for training
        random.Random(f"{shuffle_seed}:{row_id}").shuffle(field["choices"])
    values = list(choices_for(field))
    if not 1 <= len(values) <= MAX_CHOICES:
        raise ValueError(f"at most {MAX_CHOICES} options per question, got {len(values)}")
    fields = [{"name": name, "description": field["description"],
               "choices": [{"code": code, "value": value,
                            **({"description": field["choice_descriptions"][choice_key(value)]}
                               if choice_key(value) in field.get("choice_descriptions", {}) else {})}
                           for code, value in zip(string.ascii_uppercase, values)]}]
    head = "Image: " + IMAGE_BLOCK + "\n\n" if image is not None else ""      # no image: the same prompt without the image block
    content = head + safe_json({"context": context, "schema": fields}) + "\n\nRequested field: " + MARKER
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": content}]
    template = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    start, end = template.rsplit(MARKER, 1)
    prompt = start + safe_json(name) + end
    ids = tok.encode(prompt, add_special_tokens=False)
    codes = []
    for code in string.ascii_uppercase[:len(values)]:
        combined = tok.encode(prompt + code, add_special_tokens=False)
        suffix = combined[len(ids):]
        if combined[:len(ids)] != ids or len(suffix) != 1 or suffix[0] in tok.all_special_ids:
            raise ValueError(f"choice code {code} is not one ordinary token at the answer boundary")
        codes.append(suffix[0])
    if len(set(codes)) != len(codes):
        raise ValueError("choice codes must have distinct token ids")
    if image is None:
        return {"prompt_token_ids": ids, "candidate_token_ids": codes, "choices": values, "mm": [0] * len(ids),
                "pixels": None, "grid": None, "image_tokens": 0, "field": name}
    pad = tok.convert_tokens_to_ids(IMAGE_PAD)
    where = [i for i, t in enumerate(ids) if t == pad]
    if len(where) != 1:
        raise ValueError(f"expected one image placeholder, found {len(where)}")
    pixels, grid = image
    n = image_tokens(grid)
    ids = ids[:where[0]] + [pad] * n + ids[where[0] + 1:]
    return {"prompt_token_ids": ids, "candidate_token_ids": codes, "choices": values,
            "mm": [1 if t == pad else 0 for t in ids], "pixels": pixels, "grid": grid,
            "image_tokens": n, "field": name}


def runtime_row(enc, row_id, family, kind, target=None):
    """The row nimble's evaluate() and CandidateTrainer read: ids, candidates, label, and the multimodal inputs."""
    row = {"id": row_id, "family": family, "kind": kind, "input_ids": enc["prompt_token_ids"],
           "candidate_ids": enc["candidate_token_ids"], "choices": enc["choices"], "mm": enc["mm"],
           "pixels": enc["pixels"], "grid": enc["grid"], "target": target}
    if target is not None:
        keys = [choice_key(v) for v in enc["choices"]]
        row["labels"] = keys.index(choice_key(target))
    return row


class MMCollator:
    """nimble's CandidateCollator plus the multimodal inputs (left padded, like the text tokens)."""

    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, rows):
        L = max(len(r["input_ids"]) for r in rows)
        pad = lambda seq, value: [value] * (L - len(seq)) + list(seq)
        batch = {
            "input_ids": torch.tensor([pad(r["input_ids"], self.pad_id) for r in rows]),
            "attention_mask": torch.tensor([[0] * (L - len(r["input_ids"])) + [1] * len(r["input_ids"]) for r in rows]),
            "candidate_ids": torch.tensor([r["candidate_ids"] + [0] * (MAX_CHOICES - len(r["candidate_ids"])) for r in rows]),
            "candidate_mask": torch.tensor([[True] * len(r["candidate_ids"]) + [False] * (MAX_CHOICES - len(r["candidate_ids"])) for r in rows]),
        }
        if any(r["pixels"] is not None for r in rows):
            batch["mm_token_type_ids"] = torch.tensor([pad(r["mm"], 0) for r in rows])
            batch["pixel_values"] = torch.cat([r["pixels"] for r in rows if r["pixels"] is not None])
            batch["image_grid_thw"] = torch.tensor([list(r["grid"]) for r in rows if r["grid"] is not None])
        if "labels" in rows[0]:
            batch["labels"] = torch.tensor([r["labels"] for r in rows])
        return batch


def candidate_logits_mm(model, inputs):
    """nimble's candidate_logits, passing the image inputs through when the batch has them."""
    extra = {k: inputs[k] for k in ("pixel_values", "image_grid_thw", "mm_token_type_ids") if k in inputs}
    logits = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], use_cache=False,
                   logits_to_keep=1, **extra).logits[:, -1, :].float()
    selected = logits.gather(1, inputs["candidate_ids"])
    return selected.masked_fill(~inputs["candidate_mask"], -torch.inf)


def install(schema_train_module):
    """Point nimble's trainer and evaluation at the multimodal scorer and collator (their code is unchanged)."""
    schema_train_module.candidate_logits = candidate_logits_mm
    schema_train_module.CandidateCollator = MMCollator


def image_tokens_of(enc):
    return enc["image_tokens"]


INSTRUCTIONS = "这条广告属于以下哪个行业类目？"
OOC = "以上都不是"
OOC_DESCRIPTION = "广告所属行业不在以上候选类目中"
DATA = ROOT / "kev_data"


def load_mm_split(name):
    """kev_data/<name>.jsonl (e.g. validation_pos_start) -> rows with the state without its image, the image file, the
    option descriptions nimble's schema needs, and the gold option name."""
    import json
    rows = []
    for line in (DATA / f"{name}.jsonl").open(encoding="utf-8"):
        r = json.loads(line)
        state = dict(r["state"])
        image_rel = state.pop("image")
        q = r["questions"]["industry"]
        criteria = {opt: (desc if isinstance(desc, str) and desc.strip() else (OOC_DESCRIPTION if opt == OOC else opt))
                    for opt, desc in q["criteria"].items()}
        rows.append({"id": r["id"], "state": state, "image_path": str(DATA / image_rel),
                     "criteria": criteria, "label": q["label"]})
    return rows


def schema_for(row):
    """nimble's model input for one row: (context, schema) from adapt_input, the same mapping as the text runs."""
    return adapt_input({"state": row["state"], "questions": {"decision": {
        "type": "choice", "instructions": INSTRUCTIONS, "criteria": row["criteria"]}}})
