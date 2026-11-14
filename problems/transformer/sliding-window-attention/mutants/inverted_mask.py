# BUG: window_mask returns the opposite convention, True where attention is hidden; the attention function uses it consistently, so only the mask itself is wrong.
import math

import torch


def window_mask(T: int, window: int) -> torch.Tensor:
    i = torch.arange(T)[:, None]
    j = torch.arange(T)[None, :]
    return (j > i) | (j <= i - window)


def sliding_window_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, window: int
) -> torch.Tensor:
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    hidden = window_mask(q.shape[-2], window)
    scores = scores.masked_fill(hidden, float("-inf"))
    return torch.softmax(scores, dim=-1) @ v
