# BUG: averages the squares over every axis except the first instead of over the last axis only, so tokens in a sequence share one scale.
import torch


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    dims = tuple(range(1, x.ndim)) if x.ndim > 1 else (-1,)
    s = torch.clamp(x.abs().amax(dim=dims, keepdim=True), min=1.0)
    u = x / s
    mean_sq = u.square().mean(dim=dims, keepdim=True)
    return u / torch.sqrt(mean_sq + eps / (s * s)) * weight
