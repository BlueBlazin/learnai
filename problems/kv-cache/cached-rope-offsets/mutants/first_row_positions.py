# BUG: rotates every batch row with the positions of row 0 (code written for a shared (T,) position vector).
import torch

from whetstone.kit import ops


def rope_positions(cache_lengths: torch.Tensor, n_new: int) -> torch.Tensor:
    return cache_lengths[:, None] + torch.arange(n_new)


def rotate_new(
    q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor, theta: float
) -> tuple[torch.Tensor, torch.Tensor]:
    return ops.rope(q, positions[0], theta), ops.rope(k, positions[0], theta)
