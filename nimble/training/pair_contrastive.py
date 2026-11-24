"""Length-bucketed contrastive pair batch sampler and pairwise decision-flip loss."""

import random
from collections import defaultdict

import torch
import torch.nn.functional as F


class PairBatchSampler:
    """Co-batches contrastive (base, counterfactual) pairs with length bucketing.

    Every micro-batch places sibling pairs at consecutive positions (2i, 2i+1) so:
      1. 99% shared context background gradients cancel within the exact same step.
      2. Nearly equal-length pairs eliminate padding waste and left-padding drift.
      3. Pairwise decision-flip losses can directly compare logits[2i] vs logits[2i+1].
    """

    def __init__(self, dataset, batch_size, seed=17, bucket_pairs=32):
        if batch_size < 2 or batch_size % 2 != 0:
            raise ValueError(f"PairBatchSampler requires an even batch_size >= 2, got {batch_size}")
        self.dataset = dataset
        self.batch_size = batch_size
        self.pairs_per_batch = batch_size // 2
        self.seed = seed
        self.bucket_pairs = max(bucket_pairs, self.pairs_per_batch)
        self._epoch = 0

        by_pair = defaultdict(list)
        singletons = []
        for idx in range(len(dataset)):
            rec = dataset[idx]
            pid = rec.get("pair_id")
            if pid and not rec.get("is_aux", False):
                by_pair[pid].append(idx)
            else:
                singletons.append(idx)

        self.pair_groups = []
        for pid in sorted(by_pair):
            idxs = by_pair[pid]
            if len(idxs) == 2:
                self.pair_groups.append(tuple(idxs))
            else:
                singletons.extend(idxs)
        self.singletons = singletons

    def _pair_length(self, group):
        return max(len(self.dataset[i]["input_ids"]) for i in group)

    def __iter__(self):
        rng = random.Random(self.seed + self._epoch * 9973)
        self._epoch += 1

        pairs = list(self.pair_groups)
        rng.shuffle(pairs)

        batches = []
        for start in range(0, len(pairs), self.bucket_pairs):
            bucket = pairs[start:start + self.bucket_pairs]
            bucket.sort(key=self._pair_length)
            for b_start in range(0, len(bucket), self.pairs_per_batch):
                chunk = bucket[b_start:b_start + self.pairs_per_batch]
                flat = []
                for a, b in chunk:
                    flat.append(a)
                    flat.append(b)
                batches.append(flat)

        if self.singletons:
            singles = list(self.singletons)
            rng.shuffle(singles)
            bucket_singles = self.bucket_pairs * 2
            for start in range(0, len(singles), bucket_singles):
                bucket = singles[start:start + bucket_singles]
                bucket.sort(key=lambda idx: len(self.dataset[idx]["input_ids"]))
                for b_start in range(0, len(bucket), self.batch_size):
                    batches.append(bucket[b_start:b_start + self.batch_size])

        rng.shuffle(batches)
        for batch in batches:
            yield batch

    def __len__(self):
        n_pair_batches = (len(self.pair_groups) + self.pairs_per_batch - 1) // self.pairs_per_batch
        n_single_batches = (len(self.singletons) + self.batch_size - 1) // self.batch_size if self.singletons else 0
        return n_pair_batches + n_single_batches


def pairwise_flip_loss(logits, labels, pair_group_ids, margin=2.0, multiclass_only=False):
    """Bidirectional log-odds flip loss over adjacent contrastive pairs (2i, 2i+1).

    For a pair (b, c) with distinct targets y_b != y_c under synchronized candidate order:
      avg_flip_margin = 0.5 * [(lp_b[y_b] - lp_b[y_c]) + (lp_c[y_c] - lp_c[y_b])]
      loss = softplus(margin - avg_flip_margin)
    """
    n = logits.shape[0]
    if n < 2:
        return logits.new_tensor(0.0)
    log_probs = F.log_softmax(logits, dim=-1)
    pair_losses = []
    for i in range(0, n - 1, 2):
        gid_a = int(pair_group_ids[i].item())
        gid_b = int(pair_group_ids[i + 1].item())
        if gid_a < 0 or gid_a != gid_b:
            continue
        ya = int(labels[i].item())
        yb = int(labels[i + 1].item())
        if ya == yb:
            continue
        if multiclass_only and int(torch.isfinite(log_probs[i]).sum().item()) <= 2:
            continue
        # Ensure both gold indices are valid (non -inf) in both siblings.
        if not (torch.isfinite(log_probs[i, yb]) and torch.isfinite(log_probs[i + 1, ya])):
            continue
        margin_a = log_probs[i, ya] - log_probs[i, yb]
        margin_b = log_probs[i + 1, yb] - log_probs[i + 1, ya]
        avg_flip = 0.5 * (margin_a + margin_b)
        pair_losses.append(F.softplus(margin - avg_flip))
    if not pair_losses:
        return logits.new_tensor(0.0)
    return torch.stack(pair_losses).mean()
