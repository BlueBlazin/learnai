# BUG: squares x directly: in float16 the square is inf once |x| exceeds 255 (the output collapses to 0) and loses its bits below about 8e-3.
import torch


def rms_norm_f16(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    mean_sq = x.square().mean(dim=-1, keepdim=True)
    return x / torch.sqrt(mean_sq + eps) * weight
