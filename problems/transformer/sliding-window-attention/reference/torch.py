import math

import torch


def window_mask(T: int, window: int) -> torch.Tensor:
    i = torch.arange(T)[:, None]  # query position, one per row
    j = torch.arange(T)[None, :]  # key position, one per column
    # Causal (j <= i) and at most `window` keys back, counting the query itself:
    # the oldest visible key is i - window + 1.
    return (j <= i) & (j > i - window)


def sliding_window_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, window: int
) -> torch.Tensor:
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])  # (B, H, T, T)
    # Hide everything outside the window before the softmax, so that the weights are
    # normalized over the visible keys only.
    visible = window_mask(q.shape[-2], window)
    scores = scores.masked_fill(~visible, float("-inf"))
    return torch.softmax(scores, dim=-1) @ v
