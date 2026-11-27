# BUG: normalize_over puts the summed axis back by indexing with dim colons and a None, which is right for dim >= 0 only; for a negative dim the tuple of colons is empty and the new axis lands in front.
import torch


def normalize_over(x: torch.Tensor, dim: int) -> torch.Tensor:
    total = x.sum(dim=dim)
    return x / total[(slice(None),) * dim + (None,)]


def last_step(x: torch.Tensor) -> torch.Tensor:
    return x[:, -1]


def column(x: torch.Tensor, j: int) -> torch.Tensor:
    return x[:, j : j + 1]
