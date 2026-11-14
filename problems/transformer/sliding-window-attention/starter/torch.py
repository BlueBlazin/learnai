import math

import torch


def window_mask(T: int, window: int) -> torch.Tensor:
    """Boolean mask of a causal sliding window.

    T: number of positions, T >= 1. window: number of keys a query may see, window >= 1.
    Returns bool, shape (T, T): entry [i, j] is True where query i may attend to key j,
    that is where i - window < j <= i.
    """
    raise NotImplementedError


def sliding_window_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, window: int
) -> torch.Tensor:
    """Scaled dot-product attention in which query i sees only keys i - window + 1 .. i.

    q, k, v: float32, shape (B, H, T, d). Per-head queries, keys, and values.
    window: number of keys a query may see, counting its own position; window >= 1.
    Returns float32, shape (B, H, T, d).
    """
    raise NotImplementedError
