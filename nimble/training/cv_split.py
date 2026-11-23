"""Deterministic source-family-disjoint K-fold CV splitter for Nimble training data."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from nimble.training.schema_data import fingerprint, read_rows


def assign_source_family_folds(rows, num_folds=5, seed=17):
    """Assign each source_family wholly to one fold, balancing size, domains, and primitives.

    Selection depends only on structural metadata (source_family, domain, question type,
    group size) and never inspects targets or model predictions.
    """
    if not isinstance(num_folds, int) or num_folds < 2:
        raise ValueError("num_folds must be an integer >= 2")
    by_family = defaultdict(list)
    for row in rows:
        by_family[row["source_family"]].append(row)
    if len(by_family) < num_folds:
        raise ValueError(f"Need at least {num_folds} source families, got {len(by_family)}")

    # Largest families first, deterministic hash tie-break.
    ordered_families = sorted(
        by_family,
        key=lambda sf: (-len(by_family[sf]), hashlib.sha256(f"{seed}:{sf}".encode()).hexdigest()),
    )

    fold_families = [[] for _ in range(num_folds)]
    fold_sizes = [0] * num_folds
    fold_domains = [Counter() for _ in range(num_folds)]
    fold_kinds = [Counter() for _ in range(num_folds)]

    total_rows = len(rows)
    total_kinds = Counter(r["input"]["questions"]["decision"]["type"] for r in rows)
    target_size = total_rows / num_folds
    target_kinds = {k: v / num_folds for k, v in total_kinds.items()}

    for sf in ordered_families:
        group = by_family[sf]
        n = len(group)
        domain = group[0]["domain"]
        kinds = Counter(r["input"]["questions"]["decision"]["type"] for r in group)

        def score_fold(f_idx):
            # Prefer folds that do not already contain this domain, then balance size & primitives.
            domain_dup = fold_domains[f_idx][domain]
            new_size = fold_sizes[f_idx] + n
            size_err = (new_size - target_size) ** 2
            kind_err = sum((fold_kinds[f_idx][k] + kinds[k] - target_kinds[k]) ** 2 for k in ("choice", "noul", "score"))
            return (domain_dup, fold_sizes[f_idx], size_err + 0.25 * kind_err, f_idx)

        best = min(range(num_folds), key=score_fold)
        fold_families[best].append(sf)
        fold_sizes[best] += n
        fold_domains[best][domain] += 1
        fold_kinds[best].update(kinds)

    family_to_fold = {}
    for f_idx, fams in enumerate(fold_families):
        for sf in fams:
            family_to_fold[sf] = f_idx

    validate_fold_partition(rows, family_to_fold, num_folds)
    return family_to_fold


def validate_fold_partition(rows, family_to_fold, num_folds):
    """Verify strict disjointness across folds: source_family, pair family, source_id, state."""
    fold_rows = [[] for _ in range(num_folds)]
    for row in rows:
        f_idx = family_to_fold[row["source_family"]]
        if not 0 <= f_idx < num_folds:
            raise ValueError(f"Invalid fold index {f_idx}")
        fold_rows[f_idx].append(row)

    if any(not fr for fr in fold_rows):
        raise ValueError("Empty fold produced")
    if sum(len(fr) for fr in fold_rows) != len(rows):
        raise ValueError("Fold union size differs from dataset size")

    for i in range(num_folds):
        sf_i = {r["source_family"] for r in fold_rows[i]}
        pair_i = {r["family"] for r in fold_rows[i]}
        src_i = {r["provenance"]["source_id"] for r in fold_rows[i]}
        state_i = {fingerprint(r["input"]["state"]) for r in fold_rows[i]}
        # Verify every contrastive pair is intact within fold i
        pair_counts = Counter(r["family"] for r in fold_rows[i])
        if any(c != 2 for c in pair_counts.values()):
            raise ValueError(f"Fold {i} contains split contrastive pairs")
        for j in range(i + 1, num_folds):
            if sf_i & {r["source_family"] for r in fold_rows[j]}:
                raise ValueError(f"source_family overlap between folds {i} and {j}")
            if pair_i & {r["family"] for r in fold_rows[j]}:
                raise ValueError(f"pair family overlap between folds {i} and {j}")
            if src_i & {r["provenance"]["source_id"] for r in fold_rows[j]}:
                raise ValueError(f"source_id overlap between folds {i} and {j}")
            if state_i & {fingerprint(r["input"]["state"]) for r in fold_rows[j]}:
                raise ValueError(f"context state overlap between folds {i} and {j}")


def summarize_folds(rows, family_to_fold, num_folds):
    """Build a human- and machine-readable summary of all folds."""
    folds = []
    for f_idx in range(num_folds):
        val = [r for r in rows if family_to_fold[r["source_family"]] == f_idx]
        tr = [r for r in rows if family_to_fold[r["source_family"]] != f_idx]
        folds.append({
            "fold": f_idx,
            "train_rows": len(tr),
            "val_rows": len(val),
            "val_pairs": len({r["family"] for r in val}),
            "val_source_families": sorted({r["source_family"] for r in val}),
            "val_domains": dict(sorted(Counter(r["domain"] for r in val).items())),
            "val_primitives": dict(sorted(Counter(r["input"]["questions"]["decision"]["type"] for r in val).items())),
        })
    return {
        "num_folds": num_folds,
        "total_rows": len(rows),
        "total_source_families": len(family_to_fold),
        "train_fingerprint": fingerprint(rows),
        "folds": folds,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path, help="Optional path to write cv_splits.json manifest")
    args = parser.parse_args()

    rows = read_rows(args.data_dir / "train.jsonl")
    family_to_fold = assign_source_family_folds(rows, num_folds=args.num_folds, seed=args.seed)
    report = summarize_folds(rows, family_to_fold, args.num_folds)
    report["seed"] = args.seed
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
