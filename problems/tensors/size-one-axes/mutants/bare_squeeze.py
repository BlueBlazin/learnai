# BUG: last_step keeps the step axis with a slice and removes it with a bare squeeze(), which also removes the batch axis when B == 1 and the feature axis when D == 1.
import torch


def normalize_over(x: torch.Tensor, dim: int) -> torch.Tensor:
    return x / x.sum(dim=dim, keepdim=True)


def last_step(x: torch.Tensor) -> torch.Tensor:
    return x[:, -1:, :].squeeze()


def column(x: torch.Tensor, j: int) -> torch.Tensor:
    return x[:, j : j + 1]
