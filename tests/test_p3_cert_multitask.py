"""Unit tests for P3 certificate-grounded auxiliary multi-task dataset and multiclass-only flip loss."""

import unittest
from collections import Counter
from pathlib import Path

from nimble.training.cert_multitask import build_cert_aux_raw_rows
from nimble.training.cv_split import assign_source_family_folds
from nimble.training.schema_data import read_rows


DATA_DIR = Path(__file__).parents[1] / "data"


class CertMultitaskTests(unittest.TestCase):
    def test_fold0_cert_aux_zero_leakage_and_balanced_labels(self):
        rows = read_rows(DATA_DIR / "train.jsonl")
        outer_eval = read_rows(DATA_DIR / "eval.jsonl")
        mapping = assign_source_family_folds(rows, num_folds=5, seed=17)
        inner_train = [r for r in rows if mapping[r["source_family"]] != 0]
        inner_val = [r for r in rows if mapping[r["source_family"]] == 0]

        aux = build_cert_aux_raw_rows(inner_train, seed=17)
        # 1,068 pairs -> 534*2 noul (1,068) + 534*3 3-way choice (1,602) = 2,670 rows
        self.assertEqual(len(aux), 2670)

        aux_fams = {r["source_family"] for r in aux}
        val_fams = {r["source_family"] for r in inner_val}
        outer_fams = {r["source_family"] for r in outer_eval}
        self.assertFalse(aux_fams & val_fams)
        self.assertFalse(aux_fams & outer_fams)

        noul_targets = Counter(r["reference"]["target"] for r in aux if r["input"]["questions"]["decision"]["type"] == "noul")
        choice_targets = Counter(r["reference"]["target"] for r in aux if r["input"]["questions"]["decision"]["type"] == "choice")
        self.assertEqual(noul_targets, Counter({True: 534, False: 534}))
        self.assertEqual(choice_targets, Counter({"supported": 534, "refuted": 534, "unknown": 534}))

    def test_multiclass_only_flip_loss_skips_binary(self):
        try:
            import torch
            from nimble.training.pair_contrastive import pairwise_flip_loss
        except ImportError:
            self.skipTest("torch not installed locally")

        logits_bin = torch.tensor([[2.0, 0.0, -torch.inf], [2.0, 0.0, -torch.inf]])
        logits_tri = torch.tensor([[2.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        labels = torch.tensor([0, 1])
        gids = torch.tensor([0, 0])

        self.assertEqual(pairwise_flip_loss(logits_bin, labels, gids, margin=1.5, multiclass_only=True).item(), 0.0)
        self.assertGreater(pairwise_flip_loss(logits_tri, labels, gids, margin=1.5, multiclass_only=True).item(), 0.0)


if __name__ == "__main__":
    unittest.main()
