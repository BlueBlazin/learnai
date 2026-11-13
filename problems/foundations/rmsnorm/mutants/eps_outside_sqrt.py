# BUG: adds eps to the root mean square (x / (rms + eps)) instead of to the mean square inside the square root; small activations are scaled by the wrong amount.
import torch


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    s = torch.clamp(x.abs().amax(dim=-1, keepdim=True), min=1.0)
    u = x / s
    mean_sq = u.square().mean(dim=-1, keepdim=True)
    return u / (torch.sqrt(mean_sq) + eps / s) * weight
