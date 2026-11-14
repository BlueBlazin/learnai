# BUG: compares the cumulative sums with the raw uniform, assuming the last one is exactly 1; when a row sums to less than u, no entry exceeds it and the result is V, an index past the end.
import torch


def sample(probs: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    cdf = torch.cumsum(probs, dim=-1)
    return (cdf <= u[:, None]).sum(dim=-1)
