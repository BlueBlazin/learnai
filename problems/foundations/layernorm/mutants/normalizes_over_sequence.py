# BUG: the mean and variance are taken over the last two axes, so the tokens of a sequence share statistics instead of each vector having its own.
import torch


def layer_norm(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5
) -> torch.Tensor:
    dims = (-2, -1) if x.dim() >= 2 else (-1,)
    centered = x - x.mean(dim=dims, keepdim=True)
    var = centered.square().mean(dim=dims, keepdim=True)
    return centered / torch.sqrt(var + eps) * weight + bias
