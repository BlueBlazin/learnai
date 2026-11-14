# BUG: uses the first row's uniform number for every row, so the rows of a batch are not sampled independently.
import torch


def sample(probs: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    cdf = torch.cumsum(probs, dim=-1)
    threshold = u[0] * cdf[:, -1:]
    return (cdf <= threshold).sum(dim=-1)
