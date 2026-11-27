# BUG: column takes row j and turns it into a column, so it returns x[j, v] with shape (V, 1), which is the shape of (B, 1) when B == V.
import torch


def normalize_over(x: torch.Tensor, dim: int) -> torch.Tensor:
    return x / x.sum(dim=dim, keepdim=True)


def last_step(x: torch.Tensor) -> torch.Tensor:
    return x[:, -1]


def column(x: torch.Tensor, j: int) -> torch.Tensor:
    return x[j][:, None]
