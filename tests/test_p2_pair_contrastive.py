"""Unit tests for P2 PairBatchSampler and bidirectional pairwise flip loss."""

import unittest


class PairContrastiveTests(unittest.TestCase):
    def test_pair_batch_sampler_and_flip_loss(self):
        try:
            import torch
            from nimble.training.pair_contrastive import PairBatchSampler, pairwise_flip_loss
        except ImportError:
            self.skipTest("torch not installed locally")

        # 6 contrastive pairs (12 rows) with varying lengths
        ds = []
        for p in range(6):
            L = 20 + p * 15
            ds.append({"pair_id": f"pair-{p}", "input_ids": [1] * L, "labels": 0})
            ds.append({"pair_id": f"pair-{p}", "input_ids": [1] * (L + 1), "labels": 1})

        sampler = PairBatchSampler(ds, batch_size=4, seed=17, bucket_pairs=6)
        self.assertEqual(len(sampler), 3)
        batches = list(sampler)
        self.assertEqual(len(batches), 3)
        seen = []
        for b in batches:
            self.assertEqual(len(b), 4)
            # Consecutive indices (2i, 2i+1) must belong to the same contrastive pair
            self.assertEqual(ds[b[0]]["pair_id"], ds[b[1]]["pair_id"])
            self.assertEqual(ds[b[2]]["pair_id"], ds[b[3]]["pair_id"])
            seen.extend(b)
        self.assertEqual(sorted(seen), list(range(12)))

        # Verify pairwise_flip_loss behavior: collapsed predictions have higher loss than cleanly flipped predictions
        logits_collapsed = torch.tensor([[2.0, 0.0, -torch.inf], [2.0, 0.0, -torch.inf]], requires_grad=True)
        logits_flipped = torch.tensor([[3.0, 0.0, -torch.inf], [0.0, 3.0, -torch.inf]], requires_grad=True)
        labels = torch.tensor([0, 1])
        gids = torch.tensor([0, 0])

        loss_col = pairwise_flip_loss(logits_collapsed, labels, gids, margin=2.0)
        loss_flip = pairwise_flip_loss(logits_flipped, labels, gids, margin=2.0)
        self.assertGreater(loss_col.item(), loss_flip.item())
        self.assertAlmostEqual(loss_col.item(), torch.nn.functional.softplus(torch.tensor(2.0)).item(), places=5)
        loss_col.backward()
        self.assertGreater(logits_collapsed.grad[:, :2].abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
