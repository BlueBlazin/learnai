import torch


def layer_norm(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5
) -> torch.Tensor:
    # Center each vector, then divide by its standard deviation. The variance is the
    # mean of the squared deviations (divided by d, not d - 1), with eps added inside
    # the square root.
    centered = x - x.mean(dim=-1, keepdim=True)
    var = centered.square().mean(dim=-1, keepdim=True)
    return centered / torch.sqrt(var + eps) * weight + bias
