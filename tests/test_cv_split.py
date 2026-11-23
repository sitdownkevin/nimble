"""Unit tests for source-family-disjoint K-fold CV splitting and enhanced summary metrics."""

import unittest
from pathlib import Path

from nimble.training.cv_split import assign_source_family_folds, summarize_folds, validate_fold_partition
from nimble.training.schema_data import read_rows, validate_separation


DATA_DIR = Path(__file__).parents[1] / "data"


class CVSplitTests(unittest.TestCase):
    def test_five_fold_partition_on_canonical_train(self):
        rows = read_rows(DATA_DIR / "train.jsonl")
        outer_eval = read_rows(DATA_DIR / "eval.jsonl")
        mapping = assign_source_family_folds(rows, num_folds=5, seed=17)
        validate_fold_partition(rows, mapping, 5)

        # Deterministic reproducibility
        self.assertEqual(mapping, assign_source_family_folds(rows, num_folds=5, seed=17))

        summary = summarize_folds(rows, mapping, 5)
        self.assertEqual(summary["total_rows"], 2676)
        self.assertEqual(summary["total_source_families"], 34)
        self.assertEqual(sum(f["val_rows"] for f in summary["folds"]), 2676)

        outer_families = {r["source_family"] for r in outer_eval}
        for f in summary["folds"]:
            self.assertGreaterEqual(f["val_rows"], 450)
            self.assertLessEqual(f["val_rows"], 620)
            self.assertFalse(set(f["val_source_families"]) & outer_families)
            inner_tr = [r for r in rows if mapping[r["source_family"]] != f["fold"]]
            inner_va = [{**r, "split": "validation"} for r in rows if mapping[r["source_family"]] == f["fold"]]
            validate_separation(inner_tr, inner_va)

    def test_invalid_fold_counts_rejected(self):
        rows = read_rows(DATA_DIR / "train.jsonl")
        with self.assertRaises(ValueError):
            assign_source_family_folds(rows, num_folds=1)
        with self.assertRaises(ValueError):
            assign_source_family_folds(rows, num_folds=35)


if __name__ == "__main__":
    unittest.main()
