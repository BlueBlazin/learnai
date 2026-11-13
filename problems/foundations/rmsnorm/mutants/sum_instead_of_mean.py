# BUG: sums the squares instead of averaging them, so it divides by the L2 norm and the output is too small by a factor of sqrt(d).
import torch


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    s = torch.clamp(x.abs().amax(dim=-1, keepdim=True), min=1.0)
    u = x / s
    sum_sq = u.square().sum(dim=-1, keepdim=True)
    return u / torch.sqrt(sum_sq + eps / (s * s)) * weight
