# BUG: walks the distribution from the top by using 1 - u in place of u; the samples still follow the distribution, but each u selects the wrong token.
import torch


def sample(probs: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    cdf = torch.cumsum(probs, dim=-1)
    threshold = (1.0 - u[:, None]) * cdf[:, -1:]
    return (cdf < threshold).sum(dim=-1)
