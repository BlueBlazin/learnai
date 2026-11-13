# BUG: squares x directly; for |x| above about 1.8e19 the square overflows float32 to inf and the output collapses to 0.
import torch


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    mean_sq = x.square().mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(mean_sq + eps) * weight
