# BUG: computes in float32 and casts the result back; the numbers are right, but the statement requires float16 throughout.
import torch


def rms_norm_f16(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    wide = x.float()
    mean_sq = wide.square().mean(dim=-1, keepdim=True)
    return (wide / torch.sqrt(mean_sq + eps) * weight.float()).half()
