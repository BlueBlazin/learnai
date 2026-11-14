# BUG: takes argmax of the mask `cdf > u` with the raw uniform; when a row's cumulative sums never exceed u the mask is all False and argmax returns token 0.
import torch


def sample(probs: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    cdf = torch.cumsum(probs, dim=-1)
    return (cdf > u[:, None]).int().argmax(dim=-1)
