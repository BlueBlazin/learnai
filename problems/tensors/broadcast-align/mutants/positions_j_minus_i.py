# BUG: relative_positions puts the row of positions on the left of the subtraction, so the table is j - i, the negative of what is asked.
import torch


def relative_positions(T: int) -> torch.Tensor:
    pos = torch.arange(T)
    return pos[None, :] - pos[:, None]


def scale_rows(x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    return x * s[:, None, None]


def pairwise_sq_dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    diff = a[:, None, :] - b[None, :, :]
    return (diff * diff).sum(dim=-1)
