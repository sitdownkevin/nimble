"""Zero-initialized gated hybrid readout head (pretrained letter head + option-span pointer residual)."""

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HYBRID_HEAD_FILENAME = "hybrid_head.pt"


class HybridGateHead(nn.Module):
    """Combines exact FP32 pretrained letter logits with a zero-init option-span pointer residual.

    At step 0, ``gate = 0`` so ``tanh(gate) == 0`` and the output equals the pretrained letter
    logits bit-for-bit, eliminating cold-start degradation on small datasets.
    """

    def __init__(self, hidden_size, head_dim=256, init_scale=4.0):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.head_dim = int(head_dim)
        self.q_norm = nn.LayerNorm(self.hidden_size)
        self.k_norm = nn.LayerNorm(self.hidden_size)
        self.w_pool = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.w_q = nn.Linear(self.hidden_size, self.head_dim, bias=False)
        self.w_k = nn.Linear(self.hidden_size, self.head_dim, bias=False)
        self.log_scale = nn.Parameter(torch.tensor(math.log(float(init_scale)), dtype=torch.float32))
        self.gate = nn.Parameter(torch.zeros(1, dtype=torch.float32))

    def forward(self, hidden_states, letter_logits, option_spans, candidate_mask):
        """Add gated option-span pointer residual to ``letter_logits``.

        Args:
            hidden_states: ``(B, T, D)`` sequence hidden states from the language backbone.
            letter_logits: ``(B, M)`` FP32 pretrained candidate letter logits (masked to ``-inf``).
            option_spans: ``(B, M, 2)`` inclusive-exclusive token spans ``(start, end)`` per candidate.
            candidate_mask: ``(B, M)`` boolean mask of valid candidates.
        """
        h_seq = hidden_states.float()
        h_ans = self.q_norm(h_seq[:, -1, :])
        q_vec = F.normalize(self.w_q(h_ans), dim=-1, eps=1e-6)
        q_pool = self.w_pool(h_ans)
        inv_sqrt_d = 1.0 / math.sqrt(self.hidden_size)
        scale = self.log_scale.clamp(-2.0, 4.0).exp()
        alpha = torch.tanh(self.gate)

        bsz, max_c = letter_logits.shape
        ptr_logits = letter_logits.new_zeros((bsz, max_c))
        seq_len = h_seq.shape[1]

        for b in range(bsz):
            valid_idx = torch.nonzero(candidate_mask[b], as_tuple=False).squeeze(-1)
            if valid_idx.numel() == 0:
                continue
            pooled_list = []
            for k in valid_idx.tolist():
                s = int(option_spans[b, k, 0].item())
                e = int(option_spans[b, k, 1].item())
                s = max(0, min(s, seq_len - 1))
                e = max(s + 1, min(e, seq_len))
                span_h = h_seq[b, s:e, :]
                attn = torch.softmax((span_h @ q_pool[b]) * inv_sqrt_d, dim=0)
                pooled_list.append((attn.unsqueeze(-1) * span_h).sum(dim=0))
            u = torch.stack(pooled_list, dim=0)
            if u.shape[0] > 1:
                u = u - u.mean(dim=0, keepdim=True)
            k_vec = F.normalize(self.w_k(self.k_norm(u)), dim=-1, eps=1e-6)
            scores = scale * (k_vec @ q_vec[b])
            ptr_logits[b, valid_idx] = scores

        out = letter_logits + alpha * ptr_logits
        return out.masked_fill(~candidate_mask, -torch.inf)


def save_hybrid_head(head, directory):
    path = Path(directory) / HYBRID_HEAD_FILENAME
    payload = {
        "hidden_size": head.hidden_size,
        "head_dim": head.head_dim,
        "state_dict": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
    }
    torch.save(payload, path)
    return path


def load_hybrid_head(directory, device="cpu"):
    path = Path(directory) / HYBRID_HEAD_FILENAME
    if not path.exists():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=True)
    head = HybridGateHead(payload["hidden_size"], head_dim=payload["head_dim"])
    head.load_state_dict(payload["state_dict"], strict=True)
    return head.to(device)
