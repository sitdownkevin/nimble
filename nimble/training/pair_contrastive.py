"""Contrastive pair batch sampler (length-bucketed or kind-stratified) and pairwise decision-flip loss."""

import random
from collections import defaultdict

import torch
import torch.nn.functional as F


class PairBatchSampler:
    """Co-batches contrastive (base, counterfactual) pairs.

    Every micro-batch places sibling pairs at consecutive positions (2i, 2i+1) so:
      1. 99% shared context background gradients cancel within the exact same step.
      2. Sibling pairs share nearly identical length (minimal intra-pair padding).
      3. Pairwise decision-flip losses can directly compare logits[2i] vs logits[2i+1].

    Modes:
      - ``length_bucket`` (P2 default): bucket pairs by length for max throughput.
      - ``stratified`` (P4): interleave pairs proportionally across question kinds
        (choice / noul / score) without length-clustering, preventing short noul
        pairs from concentrating into correlated high-variance batches.
    """

    def __init__(self, dataset, batch_size, seed=17, bucket_pairs=32, sampling_mode="length_bucket"):
        if batch_size < 2 or batch_size % 2 != 0:
            raise ValueError(f"PairBatchSampler requires an even batch_size >= 2, got {batch_size}")
        if sampling_mode not in ("length_bucket", "stratified"):
            raise ValueError(f"Unsupported sampling_mode: {sampling_mode}")
        self.dataset = dataset
        self.batch_size = batch_size
        self.pairs_per_batch = batch_size // 2
        self.seed = seed
        self.bucket_pairs = max(bucket_pairs, self.pairs_per_batch)
        self.sampling_mode = sampling_mode
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

    def _pair_kind(self, group):
        return self.dataset[group[0]].get("kind", "unknown")

    def _interleave_stratified(self, pairs, rng):
        by_kind = defaultdict(list)
        for g in pairs:
            by_kind[self._pair_kind(g)].append(g)
        kind_keys = sorted(by_kind.keys())
        rng.shuffle(kind_keys)
        # Assign each item in each kind a uniformly spaced position in [0, 1) plus tiny jitter
        # so all kinds are smoothly interleaved across the epoch without clustering.
        scheduled = []
        for k_idx, k in enumerate(kind_keys):
            items = by_kind[k]
            rng.shuffle(items)
            n = len(items)
            offset = rng.random() / max(n, 1)
            for i, g in enumerate(items):
                pos = (i + 0.5) / n + (offset - 0.5 / n) * 0.25 + k_idx * 1e-9
                scheduled.append((pos, g))
        scheduled.sort(key=lambda x: x[0])
        return [g for _, g in scheduled]

    def __iter__(self):
        rng = random.Random(self.seed + self._epoch * 9973)
        self._epoch += 1

        pairs = list(self.pair_groups)
        batches = []
        if self.sampling_mode == "stratified":
            ordered = self._interleave_stratified(pairs, rng)
            for b_start in range(0, len(ordered), self.pairs_per_batch):
                chunk = ordered[b_start:b_start + self.pairs_per_batch]
                flat = []
                for a, b in chunk:
                    flat.append(a)
                    flat.append(b)
                batches.append(flat)
        else:
            rng.shuffle(pairs)
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
                if self.sampling_mode == "length_bucket":
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
        if not (torch.isfinite(log_probs[i, yb]) and torch.isfinite(log_probs[i + 1, ya])):
            continue
        margin_a = log_probs[i, ya] - log_probs[i, yb]
        margin_b = log_probs[i + 1, yb] - log_probs[i + 1, ya]
        avg_flip = 0.5 * (margin_a + margin_b)
        pair_losses.append(F.softplus(margin - avg_flip))
    if not pair_losses:
        return logits.new_tensor(0.0)
    return torch.stack(pair_losses).mean()
