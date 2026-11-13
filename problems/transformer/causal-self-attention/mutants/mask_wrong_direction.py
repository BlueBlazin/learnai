# BUG: the mask is flipped: it hides the past (j < i) and lets every position see the future.
import math

import torch


def causal_self_attention(
    x: torch.Tensor,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    w_o: torch.Tensor,
) -> torch.Tensor:
    seq = x.shape[1]
    q, k, v = x @ w_q.T, x @ w_k.T, x @ w_v.T
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    past = torch.tril(torch.ones(seq, seq, dtype=torch.bool), diagonal=-1)
    scores = scores.masked_fill(past, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    return (attn @ v) @ w_o.T
