"""Certificate-grounded auxiliary multi-task dataset builder (strictly in-dataset, zero external data).

Extracts dense Hop-1 multi-sentence join & evidence-necessity supervision directly from the
`evidence_certificate` embedded in Nimble's canonical `data/train.jsonl` training rows:
  1. `cert_focus_noul`: minimal-pair binary verification (base -> True, counterfactual -> False)
     on each pair's focus atomic proposition.
  2. `cert_focus_3way_choice`: 3-way entailment/refutation/abstention choice task
     {supported, refuted, unknown} using `base`, `counterfactual`, and deterministically
     reconstructed `remove_left`/`remove_right` missing-evidence contexts.
"""

import hashlib
from collections import defaultdict

from nimble.datasets.contrastive_data import apply_edit
from nimble.training.schema_data import as_scoring, encode_scoring, runtime_record


THREE_WAY_CRITERIA = {
    "supported": "The context contains sufficient factual evidence confirming that the statement is true.",
    "refuted": "The context contains factual evidence showing that the statement is false.",
    "unknown": "The context lacks sufficient evidence to determine whether the statement is true or false.",
}

NOUL_CRITERIA = {
    "false": "The context refutes the factual statement.",
    "true": "The context supports the factual statement.",
}


def _reconstruct_missing_state(base_state, spec, remove_side):
    spans = spec["focus_evidence"]
    span = spans[0] if remove_side == "left" else spans[1]
    return apply_edit(base_state, {"path": span["path"], "old": span["text"], "new": ""})


def _make_noul_aux_row(raw_row, focus_statement, target_bool, suffix):
    return {
        "id": f"aux-noul-{raw_row['id']}-{suffix}",
        "family": f"aux-noul-{raw_row['family']}",
        "source_family": raw_row["source_family"],
        "split": "train",
        "domain": raw_row["domain"],
        "input": {
            "state": raw_row["input"]["state"],
            "questions": {
                "decision": {
                    "type": "noul",
                    "instructions": (
                        "Based only on the facts in the context, is the following statement true? "
                        f"Statement: {focus_statement}"
                    ),
                    "criteria": NOUL_CRITERIA,
                }
            },
        },
        "reference": {"target": bool(target_bool)},
    }


def _make_3way_choice_aux_row(raw_row, state, focus_statement, target_label, suffix):
    return {
        "id": f"aux-3way-{raw_row['family']}-{suffix}",
        "family": f"aux-3way-{raw_row['family']}",
        "source_family": raw_row["source_family"],
        "split": "train",
        "domain": raw_row["domain"],
        "input": {
            "state": state,
            "questions": {
                "decision": {
                    "type": "choice",
                    "instructions": (
                        "Using only the facts in the context (without guessing), how does the context "
                        f"bear on the following statement? Statement: {focus_statement}"
                    ),
                    "criteria": THREE_WAY_CRITERIA,
                }
            },
        },
        "reference": {"target": target_label},
    }


def build_cert_aux_raw_rows(raw_train_rows, seed=17, aux_ratio=1.0):
    """Build balanced certificate-grounded auxiliary raw rows strictly from `raw_train_rows`.

    Up to `aux_ratio` of training pairs contribute to one stream:
      - 50% of selected pairs -> `cert_focus_noul` minimal pair (base -> True, cf -> False)
      - 50% of selected pairs -> `cert_focus_3way_choice` triplet (base -> supported, cf -> refuted,
        reconstructed remove_left/remove_right -> unknown)
    """
    if not 0.0 < aux_ratio <= 1.0:
        raise ValueError(f"aux_ratio must be in (0, 1], got {aux_ratio}")
    by_pair = defaultdict(dict)
    for row in raw_train_rows:
        if "evidence_certificate" not in row:
            continue
        by_pair[row["family"]][row["variant"]] = row

    ordered_pairs = sorted(
        (pid for pid, pair in by_pair.items() if "base" in pair and "counterfactual" in pair),
        key=lambda pid: hashlib.sha256(f"{seed}:{pid}".encode()).hexdigest(),
    )
    if aux_ratio < 1.0:
        keep_n = max(2, int(round(len(ordered_pairs) * aux_ratio)))
        ordered_pairs = ordered_pairs[:keep_n]

    aux_rows = []
    half = len(ordered_pairs) // 2
    for idx, pid in enumerate(ordered_pairs):
        pair = by_pair[pid]
        base_row = pair["base"]
        cf_row = pair["counterfactual"]
        spec = base_row["evidence_certificate"]["spec"]
        focus_id = spec["focus_atom"]
        focus_stmt = next(a["statement"] for a in spec["atoms"] if a["id"] == focus_id)

        if idx < half:
            aux_rows.append(_make_noul_aux_row(base_row, focus_stmt, True, "base"))
            aux_rows.append(_make_noul_aux_row(cf_row, focus_stmt, False, "cf"))
        else:
            side = "left" if idx % 2 == 0 else "right"
            missing_state = _reconstruct_missing_state(base_row["input"]["state"], spec, side)
            aux_rows.append(_make_3way_choice_aux_row(base_row, base_row["input"]["state"], focus_stmt, "supported", "base"))
            aux_rows.append(_make_3way_choice_aux_row(cf_row, cf_row["input"]["state"], focus_stmt, "refuted", "cf"))
            aux_rows.append(_make_3way_choice_aux_row(base_row, missing_state, focus_stmt, "unknown", f"rem_{side}"))
    return aux_rows


def encode_cert_aux_records(raw_train_rows, tokenizer, max_length, seed=17, aux_weight=0.5, aux_ratio=1.0):
    """Encode certificate auxiliary rows into runtime records (skipping any overlength rows)."""
    aux_raw = build_cert_aux_raw_rows(raw_train_rows, seed=seed, aux_ratio=aux_ratio)
    encoded_records = []
    kept_raw = []
    skipped_overlength = 0
    for row in aux_raw:
        try:
            enc = encode_scoring(
                as_scoring(row, True),
                tokenizer,
                max_length,
                shuffle_seed=seed,
                preserve_ordered_kinds=True,
                sync_pair_shuffle=True,
            )
        except ValueError as exc:
            if "Longest prompt has" in str(exc):
                skipped_overlength += 1
                continue
            raise
        rt = runtime_record(enc, row)
        rt["is_aux"] = True
        rt["sample_weight"] = float(aux_weight)
        encoded_records.append(rt)
        kept_raw.append(row)
    return encoded_records, kept_raw, skipped_overlength
