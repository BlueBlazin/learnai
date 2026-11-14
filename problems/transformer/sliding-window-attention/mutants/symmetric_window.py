# BUG: uses the distance |i - j| < window and forgets the causal rule, so each query also sees the window - 1 keys after it.
import math

import torch


def window_mask(T: int, window: int) -> torch.Tensor:
    i = torch.arange(T)[:, None]
    j = torch.arange(T)[None, :]
    return (j - i).abs() < window


def sliding_window_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, window: int
) -> torch.Tensor:
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    visible = window_mask(q.shape[-2], window)
    scores = scores.masked_fill(~visible, float("-inf"))
    return torch.softmax(scores, dim=-1) @ v
