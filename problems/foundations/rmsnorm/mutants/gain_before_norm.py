# BUG: multiplies by the gain before normalizing instead of after, so the gain's overall scale is normalized away.
import torch


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x = x * weight
    s = torch.clamp(x.abs().amax(dim=-1, keepdim=True), min=1.0)
    u = x / s
    mean_sq = u.square().mean(dim=-1, keepdim=True)
    return u / torch.sqrt(mean_sq + eps / (s * s))
