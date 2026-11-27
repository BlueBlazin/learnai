# BUG: pairwise_sq_dist sums the (N, M, D) differences over axis 1 (the points of b) instead of the feature axis, which returns shape (N, D).
import torch


def relative_positions(T: int) -> torch.Tensor:
    pos = torch.arange(T)
    return pos[:, None] - pos[None, :]


def scale_rows(x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    return x * s[:, None, None]


def pairwise_sq_dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    diff = a[:, None, :] - b[None, :, :]
    return (diff * diff).sum(dim=1)
