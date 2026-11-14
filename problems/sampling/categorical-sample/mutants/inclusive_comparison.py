# BUG: takes the first index whose cumulative sum is >= the threshold instead of > it, so u = 0 returns token 0 even when its probability is 0, and a u exactly on a boundary returns the token before it.
import torch


def sample(probs: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    cdf = torch.cumsum(probs, dim=-1)
    threshold = u[:, None] * cdf[:, -1:]
    return (cdf < threshold).sum(dim=-1)
