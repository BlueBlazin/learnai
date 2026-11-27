# BUG: pairwise_sq_dist puts b on the row axis and a on the column axis, so it returns the (M, N) table: the transpose of what is asked.
import torch


def relative_positions(T: int) -> torch.Tensor:
    pos = torch.arange(T)
    return pos[:, None] - pos[None, :]


def scale_rows(x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    return x * s[:, None, None]


def pairwise_sq_dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    diff = a[None, :, :] - b[:, None, :]
    return (diff * diff).sum(dim=-1)
