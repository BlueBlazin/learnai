import math

import torch


def qk_norm_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Causal attention with RMS-normalized queries and keys (QK-norm), then RoPE.

    q, k, v: float32, shape (B, H, T, d), d even. Per-head queries, keys, and values.
    q_weight, k_weight: float32, shape (d,). Gains of the query norm and the key norm.
    cos, sin: float32, shape (T, d/2). cos[t, i] and sin[t, i] of the RoPE angle of pair i
        (dimensions 2i and 2i + 1) at the position of token t.
    eps: positive float, added to the mean square inside the square root.
    Returns float32, shape (B, H, T, d).
    """
    raise NotImplementedError
