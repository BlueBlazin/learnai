# BUG: the bias is added before the gain is applied, so the bias gets scaled by the gain.
import torch


def layer_norm(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5
) -> torch.Tensor:
    centered = x - x.mean(dim=-1, keepdim=True)
    var = centered.square().mean(dim=-1, keepdim=True)
    return (centered / torch.sqrt(var + eps) + bias) * weight
