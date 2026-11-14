import math

import torch


def gqa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Scaled dot-product attention in which groups of query heads share one key/value head.

    q: float32, shape (B, H, T, d). Queries, one set per query head.
    k, v: float32, shape (B, H_kv, S, d). Keys and values; H is a multiple of H_kv.
    mask: bool, shape (T, S). True where query position t may attend to key position s.
    Query head h uses key/value head h // (H // H_kv).
    Returns float32, shape (B, H, T, d).
    """
    raise NotImplementedError
