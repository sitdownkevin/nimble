"""Native multimodal schema-conditioned candidate classification for Qwen3.5.

Integrates Qwen3.5 vision tokens (3D M-RoPE + merged patch embeddings) with Nimble's SOTA stack:
- Zero-duplication ``sandwich_v3`` multimodal layout (Task & Policy -> Image + Clean Context -> Full Choices)
- Backward-compatible ``v1`` multimodal layout for controlled ablation
- Dynamic per-epoch choice shuffling (with ordered-kind preservation and cyclic shift TTA)
- 4 Vision Tower adaptation regimes: ``frozen``, ``merger_lora``, ``merger_full``, ``lora_all``
- Zero-leakage validation-to-test temperature scaling and K-fold validation OOF calibration
"""

import hashlib
import math
import random
import string

import torch

from nimble.scoring.parallel_schema import (
    SYSTEM_PROMPT,
    _format_choice_value,
    _sanitize_control_tokens,
    choice_key,
    choices_for,
    render_clean_context,
    safe_json,
    validate_schema,
)
from nimble.training.deltanet_ema import _recalibrate_row, fit_temperature_1d
from nimble.vision import image_inputs, image_tokens

IMAGE_BLOCK = "<|vision_start|><|image_pad|><|vision_end|>"
IMAGE_PAD = "<|image_pad|>"
MARKER = "__PARALLEL_FIELD_TARGET__"
MAX_CHOICES = 26
VISION_MODES = ("frozen", "merger_lora", "merger_full", "lora_all")


def image_for(source):
    """Return ``(pixel_values_bf16, grid_thw)`` for an image path or data URI (cached in CPU RAM)."""
    pixels, grid = image_inputs(str(source))
    return pixels.to(torch.bfloat16), tuple(grid)


def render_mm_sandwich_v3_content(context, fields, marker, has_image=True, raw_json_context=False):
    """Render multimodal ``sandwich_v3`` user content.

    Layout:
      ## Task & Policy -> ## Context (Image block + de-escaped clean state) -> ## Choices -> Requested field
    """
    parts = ["## Task & Policy\n"]
    for field in fields:
        parts.append(f"Field: {_sanitize_control_tokens(str(field['name']))}\n")
        parts.append(f"Instructions: {_sanitize_control_tokens(str(field['description']))}\n")
    parts.append("\n## Context\n")
    clean_ctx = (
        (safe_json(context) if raw_json_context else render_clean_context(context)).strip()
        if isinstance(context, str) and context.strip()
        else ""
    )
    if has_image and clean_ctx:
        parts.append(IMAGE_BLOCK + "\n" + clean_ctx)
    elif has_image:
        parts.append(IMAGE_BLOCK)
    elif clean_ctx:
        parts.append(clean_ctx)
    else:
        parts.append("(no text context)")
    parts.append("\n\n## Choices\n")
    for field in fields:
        parts.append(f"Field: {_sanitize_control_tokens(str(field['name']))}\n")
        for ch in field["choices"]:
            code = ch["code"]
            val = _sanitize_control_tokens(_format_choice_value(ch["value"]))
            desc = ch.get("description")
            line = f"- {code} ({val}): {_sanitize_control_tokens(str(desc))}" if desc else f"- {code}: {val}"
            parts.append(line + "\n")
    parts.append("\nRequested field: " + marker)
    return "".join(parts)


def encode_mm_one(
    tokenizer,
    context,
    schema,
    image=None,
    prompt_format="sandwich_v3",
    shuffle_seed=None,
    row_id="",
    kind="choice",
    preserve_ordered_kinds=True,
    cyclic_shift=0,
    max_length=2048,
):
    """Encode a single multimodal (or text-only) bounded decision row."""
    validate_schema(schema)
    names = list(schema)
    if len(names) != 1:
        raise ValueError(f"Expected 1 decision field per prompt, got {len(names)}")
    if prompt_format not in ("v1", "sandwich_v3", "sandwich_v3_rawjson"):
        raise ValueError(f"Unsupported multimodal prompt_format: {prompt_format}")

    name = names[0]
    field = dict(schema[name])
    values = list(choices_for(field))
    if not 1 <= len(values) <= MAX_CHOICES:
        raise ValueError(f"At most {MAX_CHOICES} options allowed, got {len(values)}")

    should_shuffle = shuffle_seed is not None and not (preserve_ordered_kinds and kind in ("noul", "score"))
    if should_shuffle:
        random.Random(f"{shuffle_seed}:{row_id}").shuffle(values)
    if cyclic_shift and len(values) > 1:
        s = int(cyclic_shift) % len(values)
        if s:
            values = values[s:] + values[:s]

    fields = [{
        "name": name,
        "description": field["description"],
        "choices": [
            {
                "code": code,
                "value": value,
                **({"description": field["choice_descriptions"][choice_key(value)]}
                   if choice_key(value) in field.get("choice_descriptions", {}) else {}),
            }
            for code, value in zip(string.ascii_uppercase, values)
        ],
    }]

    has_image = image is not None
    if prompt_format in ("sandwich_v3", "sandwich_v3_rawjson"):
        content = render_mm_sandwich_v3_content(
            context, fields, MARKER,
            has_image=has_image,
            raw_json_context=(prompt_format == "sandwich_v3_rawjson"),
        )
    else:
        ctx_str = context if (isinstance(context, str) and context.strip()) else "{}"
        head = "Image: " + IMAGE_BLOCK + "\n\n" if has_image else ""
        content = head + safe_json({"context": ctx_str, "schema": fields}) + "\n\nRequested field: " + MARKER

    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": content}]
    template = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    start, end = template.rsplit(MARKER, 1)
    prompt = start + safe_json(name) + end
    ids = tokenizer.encode(prompt, add_special_tokens=False)

    codes = []
    for code in string.ascii_uppercase[:len(values)]:
        combined = tokenizer.encode(prompt + code, add_special_tokens=False)
        suffix = combined[len(ids):]
        if combined[:len(ids)] != ids or len(suffix) != 1 or suffix[0] in tokenizer.all_special_ids:
            raise ValueError(f"Choice code {code} is not a single ordinary token at the answer boundary")
        codes.append(suffix[0])
    if len(set(codes)) != len(codes):
        raise ValueError("Choice codes must have distinct token IDs")

    if not has_image:
        if len(ids) > max_length:
            raise ValueError(f"Text prompt has {len(ids)} tokens; limit is {max_length}")
        return {
            "prompt_token_ids": ids,
            "candidate_token_ids": codes,
            "choices": values,
            "mm": [0] * len(ids),
            "pixels": None,
            "grid": None,
            "image_tokens": 0,
            "field": name,
        }

    pad_tok = tokenizer.convert_tokens_to_ids(IMAGE_PAD)
    where = [i for i, t in enumerate(ids) if t == pad_tok]
    if len(where) != 1:
        raise ValueError(f"Expected exactly 1 image placeholder, found {len(where)}")
    pixels, grid = image
    n_vis = image_tokens(grid)
    ids = ids[:where[0]] + [pad_tok] * n_vis + ids[where[0] + 1:]
    if len(ids) > max_length:
        raise ValueError(f"Multimodal prompt has {len(ids)} tokens (vis={n_vis}); limit is {max_length}")
    return {
        "prompt_token_ids": ids,
        "candidate_token_ids": codes,
        "choices": values,
        "mm": [1 if t == pad_tok else 0 for t in ids],
        "pixels": pixels,
        "grid": grid,
        "image_tokens": n_vis,
        "field": name,
    }


def build_mm_runtime_row(enc, row_id, family, kind, target=None, pair_id=None):
    """Convert an ``encode_mm_one`` output dict into a trainer/evaluator runtime record."""
    row = {
        "id": row_id,
        "family": family,
        "kind": kind,
        "input_ids": enc["prompt_token_ids"],
        "candidate_ids": enc["candidate_token_ids"],
        "choices": enc["choices"],
        "mm": enc["mm"],
        "pixels": enc["pixels"],
        "grid": enc["grid"],
        "image_tokens": enc["image_tokens"],
        "target": target,
    }
    if pair_id is not None:
        row["pair_id"] = pair_id
    if target is not None:
        keys = [choice_key(v) for v in enc["choices"]]
        row["labels"] = keys.index(choice_key(target))
    return row


class MMCandidateCollator:
    """Left-padding collator supporting both multimodal (3D M-RoPE + pixels) and text-only rows."""

    def __init__(self, pad_id, flip_exclude_score=False):
        self.pad_id = pad_id
        self.flip_exclude_score = flip_exclude_score

    def __call__(self, rows):
        length = max(len(r["input_ids"]) for r in rows)
        for r in rows:
            n = len(r["candidate_ids"])
            if not 1 <= n <= MAX_CHOICES or len(set(r["candidate_ids"])) != n:
                raise ValueError("Invalid candidate token set")
            if "labels" in r and not 0 <= r["labels"] < n:
                raise ValueError("Gold label points outside real candidates")

        pad_left = lambda seq, val: [val] * (length - len(seq)) + list(seq)
        batch = {
            "input_ids": torch.tensor([pad_left(r["input_ids"], self.pad_id) for r in rows], dtype=torch.long),
            "attention_mask": torch.tensor(
                [[0] * (length - len(r["input_ids"])) + [1] * len(r["input_ids"]) for r in rows],
                dtype=torch.long,
            ),
            "candidate_ids": torch.tensor(
                [r["candidate_ids"] + [0] * (MAX_CHOICES - len(r["candidate_ids"])) for r in rows],
                dtype=torch.long,
            ),
            "candidate_mask": torch.tensor(
                [[True] * len(r["candidate_ids"]) + [False] * (MAX_CHOICES - len(r["candidate_ids"])) for r in rows],
                dtype=torch.bool,
            ),
        }
        if any(r.get("pixels") is not None for r in rows):
            batch["mm_token_type_ids"] = torch.tensor(
                [pad_left(r["mm"], 0) for r in rows], dtype=torch.long
            )
            batch["pixel_values"] = torch.cat([r["pixels"] for r in rows if r.get("pixels") is not None], dim=0)
            batch["image_grid_thw"] = torch.tensor(
                [list(r["grid"]) for r in rows if r.get("grid") is not None], dtype=torch.long
            )
        if "labels" in rows[0]:
            batch["labels"] = torch.tensor([r["labels"] for r in rows], dtype=torch.long)
        if any(r.get("pair_id") for r in rows):
            pid_map = {}
            gids = []
            for r in rows:
                pid = r.get("pair_id")
                if not pid or (self.flip_exclude_score and r.get("kind") == "score"):
                    gids.append(-1)
                else:
                    if pid not in pid_map:
                        pid_map[pid] = len(pid_map)
                    gids.append(pid_map[pid])
            batch["pair_group_ids"] = torch.tensor(gids, dtype=torch.long)
        if any("sample_weight" in r for r in rows):
            batch["sample_weights"] = torch.tensor(
                [float(r.get("sample_weight", 1.0)) for r in rows], dtype=torch.float32
            )
        return batch


class DynamicMMDataset(torch.utils.data.Dataset):
    """Per-epoch dynamic option reshuffling for multimodal or text-only items."""

    def __init__(
        self,
        raw_items,
        tokenizer,
        max_length,
        base_seed,
        prompt_format="sandwich_v3",
        preserve_ordered_kinds=True,
        dynamic_shuffle=True,
    ):
        self.raw_items = list(raw_items)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.base_seed = base_seed
        self.prompt_format = prompt_format
        self.preserve_ordered_kinds = preserve_ordered_kinds
        self.dynamic_shuffle = dynamic_shuffle
        self.records = [None] * len(self.raw_items)
        self.current_epoch = -1
        self.set_epoch(0)

    def set_epoch(self, epoch):
        if epoch == self.current_epoch:
            return
        if epoch > 0 and not self.dynamic_shuffle:
            self.current_epoch = epoch
            return
        self.current_epoch = epoch
        seed_str = f"{self.base_seed}:ep{epoch}" if self.dynamic_shuffle else str(self.base_seed)
        for idx, item in enumerate(self.raw_items):
            if epoch > 0 and self.preserve_ordered_kinds and item["kind"] in ("noul", "score"):
                continue
            enc = encode_mm_one(
                self.tokenizer,
                item["context"],
                item["schema"],
                image=item.get("image"),
                prompt_format=self.prompt_format,
                shuffle_seed=seed_str,
                row_id=item.get("pair_id") or item["id"],
                kind=item["kind"],
                preserve_ordered_kinds=self.preserve_ordered_kinds,
                max_length=self.max_length,
            )
            self.records[idx] = build_mm_runtime_row(
                enc,
                row_id=item["id"],
                family=item["family"],
                kind=item["kind"],
                target=item["target"],
                pair_id=item.get("pair_id"),
            )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.records[idx]


def select_lora_targets(base_model, vision_mode="frozen"):
    """Return LoRA target Linear module names for the requested ``vision_mode``.

    Modes:
      - ``frozen``: LoRA on ``language_model`` Linear layers only; entire ``visual`` tower frozen.
      - ``merger_lora``: LoRA on ``language_model`` + cross-modal ``visual.merger`` Linear layers.
      - ``merger_full``: LoRA on ``language_model`` only (full ``visual.merger`` is unfrozen separately).
      - ``lora_all``: LoRA on all ``visual.blocks`` + ``visual.merger`` + ``language_model`` Linear layers.
    """
    if vision_mode not in VISION_MODES:
        raise ValueError(f"Unsupported vision_mode {vision_mode!r}; expected one of {VISION_MODES}")

    lang_targets = [
        name for name, layer in base_model.named_modules()
        if isinstance(layer, torch.nn.Linear) and ".language_model." in name
    ]
    if not lang_targets:
        lang_targets = [
            name for name, layer in base_model.named_modules()
            if isinstance(layer, torch.nn.Linear) and name.startswith("model.layers.")
        ]
    if not lang_targets:
        raise RuntimeError("No language_model Linear layers matched")

    if vision_mode in ("frozen", "merger_full"):
        return lang_targets
    if vision_mode == "merger_lora":
        vis_targets = [
            name for name, layer in base_model.named_modules()
            if isinstance(layer, torch.nn.Linear) and ".visual.merger." in name
        ]
        if not vis_targets:
            raise RuntimeError("No visual.merger Linear layers matched for merger_lora")
        return lang_targets + vis_targets

    vis_targets = [
        name for name, layer in base_model.named_modules()
        if isinstance(layer, torch.nn.Linear) and ".visual." in name
    ]
    if not vis_targets:
        raise RuntimeError("No visual Linear layers matched for lora_all")
    return lang_targets + vis_targets


def unlock_vision_merger_parameters(model):
    """Unfreeze all parameters inside ``visual.merger`` (saved/restored via ``deltanet_extra.pt``)."""
    unlocked = []
    scalars = 0
    for name, param in model.named_parameters():
        if ".visual.merger." in name and "lora_" not in name:
            param.requires_grad = True
            unlocked.append(name)
            scalars += param.numel()
    if not unlocked:
        raise RuntimeError("No visual.merger parameters found to unlock")
    return unlocked, scalars


def cyclic_shifts_for(n_choices, num_views):
    """Return ``num_views`` evenly spaced cyclic shifts in ``[0, n_choices)`` starting with ``0``."""
    if num_views <= 1 or n_choices <= 1:
        return [0]
    k = min(int(num_views), int(n_choices))
    shifts = []
    for i in range(k):
        s = int(round(i * n_choices / k)) % n_choices
        if s not in shifts:
            shifts.append(s)
    return shifts


def evaluate_kfold_oof_temperature(val_rows, summarize_fn, n_folds=5, seed=17):
    """Zero-leakage K-fold OOF scalar temperature calibration on ``validation`` rows,
    plus full-validation locked scalar temperature for downstream ``test`` evaluation.
    """
    if len(val_rows) < n_folds:
        t_full = fit_temperature_1d(val_rows)
        cal = [_recalibrate_row(r, t_full) for r in val_rows]
        return {
            "locked_temperature": t_full,
            "fold_temperatures": [t_full],
            "uncalibrated_summary": summarize_fn(val_rows),
            "oof_calibrated_summary": summarize_fn(cal),
        }

    indexed = list(enumerate(val_rows))
    indexed.sort(key=lambda pair: hashlib.sha256(f"{seed}:{pair[1]['id']}".encode()).hexdigest())
    folds = [indexed[i::n_folds] for i in range(n_folds)]
    oof_rows = [None] * len(val_rows)
    fold_temps = []
    for f_idx in range(n_folds):
        held_indices = {orig_i for orig_i, _ in folds[f_idx]}
        fit_subset = [r for i, r in enumerate(val_rows) if i not in held_indices]
        t_f = fit_temperature_1d(fit_subset)
        fold_temps.append(t_f)
        for orig_i, r in folds[f_idx]:
            oof_rows[orig_i] = _recalibrate_row(r, t_f)

    locked_t = fit_temperature_1d(val_rows)
    return {
        "locked_temperature": locked_t,
        "fold_temperatures": fold_temps,
        "uncalibrated_summary": summarize_fn(val_rows),
        "oof_calibrated_summary": summarize_fn(oof_rows),
    }


def apply_locked_temperature(rows, temperature, summarize_fn):
    """Apply a validation-fitted scalar temperature to held-out ``test`` rows (zero test leakage)."""
    recal = [_recalibrate_row(r, float(temperature)) for r in rows]
    return {
        "applied_temperature": float(temperature),
        "uncalibrated_summary": summarize_fn(rows),
        "calibrated_summary": summarize_fn(recal),
        "rows": recal,
    }
