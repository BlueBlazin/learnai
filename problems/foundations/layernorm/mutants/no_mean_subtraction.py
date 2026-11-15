# BUG: the mean is never subtracted: this is RMSNorm with a bias, not LayerNorm.
import torch


def layer_norm(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5
) -> torch.Tensor:
    mean_sq = x.square().mean(dim=-1, keepdim=True)
    return x / torch.sqrt(mean_sq + eps) * weight + bias
