import math

import torch


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    # One root mean square per vector, over the head dimension (the last axis) only.
    mean_square = x.pow(2).mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(mean_square + eps) * weight


def _rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # RoPE on adjacent pairs; cos and sin are (T, d/2) and broadcast over (B, H).
    x1, x2 = x[..., 0::2], x[..., 1::2]
    rotated = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
    return rotated.reshape(x.shape)


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
    seq, dim = q.shape[-2], q.shape[-1]
    # Normalize first, rotate second: the gains belong to the unrotated dimensions.
    # The values are neither normalized nor rotated.
    q = _rotate(_rms_norm(q, q_weight, eps), cos, sin)
    k = _rotate(_rms_norm(k, k_weight, eps), cos, sin)
    # The usual scale stays, even though the vectors are normalized.
    scores = q @ k.transpose(-2, -1) / math.sqrt(dim)
    future = torch.triu(torch.ones(seq, seq, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(future, float("-inf"))
    return torch.softmax(scores, dim=-1) @ v
