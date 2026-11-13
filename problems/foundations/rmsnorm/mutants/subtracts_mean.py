# BUG: centers x by subtracting its mean first, which is LayerNorm, not RMSNorm.
import torch


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x = x - x.mean(dim=-1, keepdim=True)
    s = torch.clamp(x.abs().amax(dim=-1, keepdim=True), min=1.0)
    u = x / s
    mean_sq = u.square().mean(dim=-1, keepdim=True)
    return u / torch.sqrt(mean_sq + eps / (s * s)) * weight
