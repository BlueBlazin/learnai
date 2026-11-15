# BUG: the float32 recipe of the rmsnorm problem, s = max(1, max|x|): it fixes the overflow but leaves small vectors unscaled, and their squares underflow float16.
import torch


def rms_norm_f16(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    s = torch.clamp(x.abs().amax(dim=-1, keepdim=True), min=1.0)
    u = x / s
    mean_sq = u.square().mean(dim=-1, keepdim=True)
    return u / torch.sqrt(mean_sq + eps / (s * s)) * weight
