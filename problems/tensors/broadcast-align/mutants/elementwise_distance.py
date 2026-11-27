# BUG: pairwise_sq_dist subtracts a and b as they are, which pairs point n of a with point n of b and returns (N,) distances instead of the (N, M) table.
import torch


def relative_positions(T: int) -> torch.Tensor:
    pos = torch.arange(T)
    return pos[:, None] - pos[None, :]


def scale_rows(x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    return x * s[:, None, None]


def pairwise_sq_dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    diff = a - b
    return (diff * diff).sum(dim=-1)
