import math

import torch


def causal_self_attention(
    x: torch.Tensor,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    w_o: torch.Tensor,
) -> torch.Tensor:
    """Single-head scaled dot-product self-attention with a causal mask.

    x: float32, shape (B, T, d). One sequence of T token vectors per batch item.
    w_q, w_k: float32, shape (d_k, d). Query and key projections, (out, in) layout.
    w_v: float32, shape (d_v, d). Value projection.
    w_o: float32, shape (d_out, d_v). Output projection.
    Returns float32, shape (B, T, d_out). Output t depends only on x[:, : t + 1].
    """
    raise NotImplementedError
