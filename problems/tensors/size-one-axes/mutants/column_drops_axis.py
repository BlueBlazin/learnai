# BUG: column indexes with the integer j, which removes the axis: it returns shape (B,) instead of (B, 1).
import torch


def normalize_over(x: torch.Tensor, dim: int) -> torch.Tensor:
    return x / x.sum(dim=dim, keepdim=True)


def last_step(x: torch.Tensor) -> torch.Tensor:
    return x[:, -1]


def column(x: torch.Tensor, j: int) -> torch.Tensor:
    return x[:, j]
