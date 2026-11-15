# BUG: eps is ignored, so a constant vector divides 0 by 0 and returns NaN, and low-variance vectors are scaled up too far.
import torch


def layer_norm(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5
) -> torch.Tensor:
    centered = x - x.mean(dim=-1, keepdim=True)
    var = centered.square().mean(dim=-1, keepdim=True)
    return centered / torch.sqrt(var) * weight + bias
