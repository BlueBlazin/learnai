# BUG: zeroes the weights outside the window after a causal softmax instead of masking the scores before it; the visible weights are not renormalized and still depend on keys outside the window.
import math

import torch


def window_mask(T: int, window: int) -> torch.Tensor:
    i = torch.arange(T)[:, None]
    j = torch.arange(T)[None, :]
    return (j <= i) & (j > i - window)


def sliding_window_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, window: int
) -> torch.Tensor:
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    seq = q.shape[-2]
    future = torch.triu(torch.ones(seq, seq, dtype=torch.bool), diagonal=1)
    attn = torch.softmax(scores.masked_fill(future, float("-inf")), dim=-1)
    return attn.masked_fill(~window_mask(seq, window), 0.0) @ v
