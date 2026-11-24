"""Validate schema-training exports and preserve the inference prompt exactly."""

import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from nimble.evaluation.evaluate_pilot import adapt_input
from nimble.scoring.parallel_schema import MODEL_ID, REVISION, choice_key, prepare_prompts


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()).hexdigest()


def read_rows(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").split("\n") if line.strip()]
    if not rows or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Records must be nonempty with unique IDs")
    return rows


def is_ordered_or_boolean(field):
    if field["type"] == "boolean":
        return True
    choices = field.get("choices", [])
    return field["type"] == "enum" and choices == [str(i) for i in range(len(choices))]


def encode_scoring(row, tokenizer, max_length, shuffle_seed=None,
                   preserve_ordered_kinds=False, sync_pair_shuffle=False, prompt_format="v1",
                   cyclic_shift=0):
    schema = json.loads(json.dumps(row["schema"]))
    field = schema[row["field"]]
    if cyclic_shift and not is_ordered_or_boolean(field):
        n_c = len(field["choices"])
        shift = cyclic_shift % n_c
        if shift:
            field["choices"] = field["choices"][shift:] + field["choices"][:shift]
    elif shuffle_seed is not None and not (preserve_ordered_kinds and is_ordered_or_boolean(field)):
        key = row["group_id"] if sync_pair_shuffle else row["id"]
        random.Random(f"{shuffle_seed}:{key}").shuffle(field["choices"])
    prompt = prepare_prompts(tokenizer, row["context"], schema, max_length, prompt_format=prompt_format)
    i = prompt.names.index(row["field"])
    keys = [choice_key(v) for v in prompt.choices[i]]
    target_index = keys.index(choice_key(row["target"]))
    out = {"id": row["id"], "group_id": row["group_id"], "source_family": row["source_family"],
           "prompt_token_ids": prompt.full_ids[i], "candidate_token_ids": prompt.candidate_ids[i],
           "target_index": target_index, "target_token_id": prompt.candidate_ids[i][target_index],
           "code_to_choice": dict(zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ", prompt.choices[i])),
           "loss": "candidate_cross_entropy_at_last_prompt_position"}
    if prompt.option_spans is not None:
        out["option_spans"] = prompt.option_spans[i]
    return out


def as_scoring(row, training):
    context, schema = adapt_input(row["input"])
    if list(schema) != ["decision"]:
        raise ValueError("Training records must contain one decision question")
    kind = row["input"]["questions"]["decision"]["type"]
    target = row["reference"]["target"]
    if kind == "score":
        if type(target) is not int:
            raise ValueError("Score gold value must be an integer level")
        target = str(target)
    elif kind == "noul" and type(target) is not bool:
        raise ValueError("Noul gold value must be a boolean")
    return {"id": row["id"], "group_id": row["family"],
            "source_family": row["source_family"] if training else row.get("source_family", row["family"]),
            "context": context, "schema": schema, "field": "decision", "target": target,
            "target_key": choice_key(target), "loss": "candidate_cross_entropy"}


def validate_separation(training, validation):
    if any(r.get("split") != "train" for r in training):
        raise ValueError("Training file contains non-training records")
    if any(r.get("split") != "validation" for r in validation):
        raise ValueError("Validation file must contain validation records only")
    if {r["source_family"] for r in training} & {r.get("source_family", r["family"]) for r in validation}:
        raise ValueError("Source family crosses training and validation")
    if {r["provenance"]["source_id"] for r in training} & {
        r.get("provenance", {}).get("source_id", r["id"]) for r in validation
    }:
        raise ValueError("Training uses a validation source")
    if {fingerprint(r["input"]["state"]) for r in training} & {fingerprint(r["input"]["state"]) for r in validation}:
        raise ValueError("Context crosses training and validation")


def runtime_record(encoded, raw):
    rec = {"input_ids": encoded["prompt_token_ids"], "candidate_ids": encoded["candidate_token_ids"],
           "labels": encoded["target_index"], "id": raw["id"],
           "kind": raw["input"]["questions"]["decision"]["type"],
           "choices": list(encoded["code_to_choice"].values()), "target": raw["reference"]["target"],
           "family": encoded["source_family"], "pair_id": raw["family"]}
    if "option_spans" in encoded:
        rec["option_spans"] = encoded["option_spans"]
    return rec


def resolve_checkpoint(model, revision):
    if revision:
        return model, revision
    if model == MODEL_ID:
        return model, REVISION
    raise ValueError('A non-default model requires an explicit --revision')


def prepare_data(directory, validation_path, tokenizer, max_length, model_id=MODEL_ID, revision=REVISION):
    directory = Path(directory)
    raw = read_rows(directory / "train.jsonl")
    manifest = json.loads((directory / "manifest.json").read_text())
    compact = manifest.get('storage_format') == 'nimble-self-contained-v1'
    if compact:
        for path, key in [(directory / 'train.jsonl', 'train.jsonl'), (Path(validation_path), 'eval.jsonl')]:
            if hashlib.sha256(path.read_bytes()).hexdigest() != manifest['file_sha256'][key]:
                raise ValueError('Frozen dataset bytes changed: ' + key)
        scoring = [as_scoring(r, True) for r in raw]
        seed = manifest['tokenization_reference']['shuffle_seed']
        tokens = [encode_scoring(r, tokenizer, max_length, seed) for r in scoring]
        token_manifest = {'model': model_id, 'revision': revision, 'shuffle_seed': seed,
                          'source_sha256': fingerprint(scoring), 'records_sha256': fingerprint(tokens)}
        reference = manifest['tokenization_reference']
        if model_id == reference['model'] and revision == reference['revision']:
            for key in ('source_sha256', 'records_sha256'):
                if token_manifest[key] != reference[key]:
                    raise ValueError('Regenerated export differs from the verified 9B export')
    else:
        scoring = read_rows(directory / "train_scoring.jsonl")
        tokens = read_rows(directory / "train_tokens.jsonl")
        token_manifest = json.loads((directory / "train_tokens.manifest.json").read_text())
    if token_manifest["model"] != model_id or token_manifest["revision"] != revision:
        raise ValueError("Token export model/revision differs")
    checks = [(fingerprint(raw), manifest["train_sha256"]),
              (fingerprint(scoring), token_manifest["source_sha256"]),
              (fingerprint(tokens), token_manifest["records_sha256"])]
    if any(a != b for a, b in checks):
        raise ValueError("Training export fingerprint differs from its manifest")
    ids = [r["id"] for r in raw]
    if ids != [r["id"] for r in scoring] or ids != [r["id"] for r in tokens]:
        raise ValueError("Training export record IDs/order differ")
    if manifest.get("pipeline_version") == "evidence-plus-reviewed-v1":
        from nimble.training.augmented_data import validate_augmented_data
        groups = validate_augmented_data(raw, manifest)
        if fingerprint(raw) != manifest['train_sha256']:
            raise ValueError('Augmented training fingerprint changed')
        if hashlib.sha256(Path(validation_path).read_bytes()).hexdigest() != manifest['eval_sha256']:
            raise ValueError('Frozen evaluation bytes changed')
    elif manifest.get("pipeline_version") == "evidence-curation-v3":
        from nimble.training.evidence_data import validate_evidence_data
        groups = validate_evidence_data(raw, manifest)
    else:
        expected = Counter(["base", "evidence_removed", "counterfactual", "paraphrase", "distractor"])
        groups = {}
        for r in raw:
            if r["quality_status"] != "model_checked" or not r["verification"]["agrees"] or not r["verification"]["unambiguous"]:
                raise ValueError("Unaccepted training example")
            groups.setdefault(r["family"], []).append(r["variant"])
        if any(Counter(v) != expected for v in groups.values()):
            raise ValueError("Training groups are incomplete")
    validation = read_rows(validation_path)
    validate_separation(raw, validation)
    train_data = []
    for r, s, t in zip(raw, scoring, tokens):
        if as_scoring(r, True) != s:
            raise ValueError(f"Scoring input or target mismatch: {r['id']}")
        regenerated = t if compact else encode_scoring(s, tokenizer, max_length, token_manifest["shuffle_seed"])
        if regenerated != t:
            raise ValueError(f"Saved tokens differ from current prompt/tokenizer: {r['id']}")
        train_data.append(runtime_record(t, r))
    val_data = [runtime_record(encode_scoring(as_scoring(r, False), tokenizer, max_length), r) for r in validation]
    audit = {"training_rows": len(raw), "training_groups": len(groups),
             "training_families": len({r['source_family'] for r in raw}),
             "validation_rows": len(validation), "token_records_reconstructed": len(tokens),
             "training_fingerprint": fingerprint(raw), "validation_fingerprint": fingerprint(validation),
             "training_tokens_fingerprint": fingerprint(tokens),
             "validation_order": "original inference choice order; no shuffling",
             "train_max_tokens": max(len(r["input_ids"]) for r in train_data),
             "validation_max_tokens": max(len(r["input_ids"]) for r in val_data),
             "training_kinds": dict(Counter(r["kind"] for r in train_data)),
             "validation_kinds": dict(Counter(r["kind"] for r in val_data)),
             "heldout_family_overlap": False, "test_split_used": False}
    return train_data, val_data, audit


def prepare_cv_data(directory, validation_path, tokenizer, max_length, num_folds, fold_index,
                    cv_seed=17, model_id=MODEL_ID, revision=REVISION,
                    preserve_ordered_kinds=False, sync_pair_shuffle=False, return_raw_train=False,
                    prompt_format="v1", return_raw_val=False):
    """Verify canonical dataset integrity, then split training rows into source-family-disjoint CV folds.

    The outer holdout (`validation_path`) is verified for separation and then discarded so it
    can never enter inner training or inner validation.
    """
    from nimble.training.cv_split import assign_source_family_folds

    if not 0 <= fold_index < num_folds:
        raise ValueError(f"fold_index must be in [0, {num_folds - 1}], got {fold_index}")
    train_data, _outer_val, base_audit = prepare_data(
        directory, validation_path, tokenizer, max_length, model_id, revision
    )
    raw = read_rows(Path(directory) / "train.jsonl")
    manifest = json.loads((Path(directory) / "manifest.json").read_text())
    shuffle_seed = manifest.get("tokenization_reference", {}).get("shuffle_seed", 17)
    family_to_fold = assign_source_family_folds(raw, num_folds=num_folds, seed=cv_seed)

    inner_train_raw, inner_val_raw = [], []
    inner_train_data = []
    for raw_row, rt_row in zip(raw, train_data):
        if family_to_fold[raw_row["source_family"]] == fold_index:
            inner_val_raw.append({**raw_row, "split": "validation"})
        else:
            inner_train_raw.append(raw_row)
            if preserve_ordered_kinds or sync_pair_shuffle or prompt_format != "v1":
                enc = encode_scoring(
                    as_scoring(raw_row, True), tokenizer, max_length,
                    shuffle_seed=shuffle_seed,
                    preserve_ordered_kinds=preserve_ordered_kinds,
                    sync_pair_shuffle=sync_pair_shuffle,
                    prompt_format=prompt_format,
                )
                inner_train_data.append(runtime_record(enc, raw_row))
            else:
                inner_train_data.append(rt_row)

    validate_separation(inner_train_raw, inner_val_raw)
    inner_val_data = [
        runtime_record(encode_scoring(as_scoring(r, False), tokenizer, max_length, prompt_format=prompt_format), r)
        for r in inner_val_raw
    ]

    val_families = sorted({r["source_family"] for r in inner_val_raw})
    audit = {
        **base_audit,
        "cv_mode": True,
        "cv_folds": num_folds,
        "cv_fold": fold_index,
        "cv_seed": cv_seed,
        "prompt_format": prompt_format,
        "preserve_ordered_kinds": preserve_ordered_kinds,
        "sync_pair_shuffle": sync_pair_shuffle,
        "outer_holdout_touched": False,
        "training_rows": len(inner_train_data),
        "training_groups": len({r["family"] for r in inner_train_raw}),
        "training_families": len({r["source_family"] for r in inner_train_raw}),
        "validation_rows": len(inner_val_data),
        "validation_groups": len({r["family"] for r in inner_val_raw}),
        "validation_families": len(val_families),
        "validation_source_families": val_families,
        "inner_training_fingerprint": fingerprint(inner_train_raw),
        "inner_validation_fingerprint": fingerprint(inner_val_raw),
        "train_max_tokens": max(len(r["input_ids"]) for r in inner_train_data),
        "validation_max_tokens": max(len(r["input_ids"]) for r in inner_val_data),
        "training_kinds": dict(Counter(r["kind"] for r in inner_train_data)),
        "validation_kinds": dict(Counter(r["kind"] for r in inner_val_data)),
    }
    if return_raw_train and return_raw_val:
        return inner_train_data, inner_val_data, audit, inner_train_raw, inner_val_raw
    if return_raw_train:
        return inner_train_data, inner_val_data, audit, inner_train_raw
    if return_raw_val:
        return inner_train_data, inner_val_data, audit, inner_val_raw
    return inner_train_data, inner_val_data, audit
