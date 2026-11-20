# BUG: returns the weighted sum of rotated rows without turning its rotary slice back by the query position, so the output depends on absolute position.
import math

import torch


def _rotate(x: torch.Tensor, positions: torch.Tensor, rope_dim: int, base: float) -> torch.Tensor:
    """Rotate the trailing `rope_dim` channels of `x` (..., N, d), row n by positions[n]."""
    freqs = base ** (-torch.arange(0, rope_dim, 2, dtype=torch.float32) / rope_dim)
    angles = positions.to(torch.float32)[:, None] * freqs  # (N, rope_dim / 2)
    cos, sin = torch.cos(angles), torch.sin(angles)
    split = x.shape[-1] - rope_dim
    plain, tail = x[..., :split], x[..., split:]
    # Interleaved pairs inside the slice: (0, 1), (2, 3), ...
    a, b = tail[..., 0::2], tail[..., 1::2]
    rotated = torch.stack((a * cos - b * sin, a * sin + b * cos), dim=-1).reshape(tail.shape)
    return torch.cat((plain, rotated), dim=-1)


def shared_kv_attention(
    q: torch.Tensor,
    kv: torch.Tensor,
    pos_q: torch.Tensor,
    pos_k: torch.Tensor,
    mask: torch.Tensor,
    rope_dim: int,
    base: float,
) -> torch.Tensor:
    q = _rotate(q, pos_q, rope_dim, base)
    kv = _rotate(kv, pos_k, rope_dim, base)  # one tensor: the key and the value
    scores = q @ kv.T / math.sqrt(q.shape[-1])  # (H, T, S)
    scores = scores.masked_fill(~mask, float("-inf"))
    out = torch.softmax(scores, dim=-1) @ kv  # (H, T, d): a mix of rotated rows
    return out
