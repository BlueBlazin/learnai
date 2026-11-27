# BUG: scale_rows multiplies by s as it is; its shape (B,) lines up with the last axis of x, so the call raises unless B == D, and then feature d is scaled by s[d].
import torch


def relative_positions(T: int) -> torch.Tensor:
    pos = torch.arange(T)
    return pos[:, None] - pos[None, :]


def scale_rows(x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    return x * s


def pairwise_sq_dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    diff = a[:, None, :] - b[None, :, :]
    return (diff * diff).sum(dim=-1)
