# BUG: the variance comes from torch.var, whose default divides by d - 1 (the unbiased estimator); LayerNorm divides by d.
import torch


def layer_norm(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5
) -> torch.Tensor:
    centered = x - x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True)
    return centered / torch.sqrt(var + eps) * weight + bias
