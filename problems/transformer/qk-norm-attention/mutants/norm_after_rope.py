# BUG: rotates first and normalizes second; the root mean square is unchanged by the rotation, but the per-dimension gains now multiply rotated dimensions.
import math

import torch


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    mean_square = x.pow(2).mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(mean_square + eps) * weight


def _rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
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
    q = _rms_norm(_rotate(q, cos, sin), q_weight, eps)
    k = _rms_norm(_rotate(k, cos, sin), k_weight, eps)
    scores = q @ k.transpose(-2, -1) / math.sqrt(dim)
    future = torch.triu(torch.ones(seq, seq, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(future, float("-inf"))
    return torch.softmax(scores, dim=-1) @ v
