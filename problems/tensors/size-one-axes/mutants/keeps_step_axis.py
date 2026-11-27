# BUG: last_step slices the last step with -1: and never removes the axis, so it returns shape (B, 1, D) instead of (B, D).
import torch


def normalize_over(x: torch.Tensor, dim: int) -> torch.Tensor:
    return x / x.sum(dim=dim, keepdim=True)


def last_step(x: torch.Tensor) -> torch.Tensor:
    return x[:, -1:]


def column(x: torch.Tensor, j: int) -> torch.Tensor:
    return x[:, j : j + 1]
