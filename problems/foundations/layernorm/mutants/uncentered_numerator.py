# BUG: the variance is right, but the numerator is x instead of x - mean, so a shift of the input leaks into the output.
import torch


def layer_norm(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5
) -> torch.Tensor:
    centered = x - x.mean(dim=-1, keepdim=True)
    var = centered.square().mean(dim=-1, keepdim=True)
    return x / torch.sqrt(var + eps) * weight + bias
