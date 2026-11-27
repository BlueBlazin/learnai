# BUG: scale_rows gives s the shape (1, B, 1), which lines up with the T axis: step t is scaled by s[t] when B == T, and the result has shape (B, B, D) when T == 1.
import torch


def relative_positions(T: int) -> torch.Tensor:
    pos = torch.arange(T)
    return pos[:, None] - pos[None, :]


def scale_rows(x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    return x * s[None, :, None]


def pairwise_sq_dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    diff = a[:, None, :] - b[None, :, :]
    return (diff * diff).sum(dim=-1)
