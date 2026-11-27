# BUG: normalize_over sums without keepdim, so the sum has lost the axis and lines up with the last axes of x: on a square input summed over its last axis, entry [i, j] is divided by the sum of row j.
import torch


def normalize_over(x: torch.Tensor, dim: int) -> torch.Tensor:
    return x / x.sum(dim=dim)


def last_step(x: torch.Tensor) -> torch.Tensor:
    return x[:, -1]


def column(x: torch.Tensor, j: int) -> torch.Tensor:
    return x[:, j : j + 1]
