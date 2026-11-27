# BUG: last_step indexes the last axis instead of the step axis, returning the last feature of every step: shape (B, T), which is the shape of (B, D) when T == D.
import torch


def normalize_over(x: torch.Tensor, dim: int) -> torch.Tensor:
    return x / x.sum(dim=dim, keepdim=True)


def last_step(x: torch.Tensor) -> torch.Tensor:
    return x[:, :, -1]


def column(x: torch.Tensor, j: int) -> torch.Tensor:
    return x[:, j : j + 1]
