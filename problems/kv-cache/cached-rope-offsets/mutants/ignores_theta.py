# BUG: ignores theta and rotates with the default base of 10000.
import torch

from whetstone.kit import ops


def rope_positions(cache_lengths: torch.Tensor, n_new: int) -> torch.Tensor:
    return cache_lengths[:, None] + torch.arange(n_new)


def rotate_new(
    q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor, theta: float
) -> tuple[torch.Tensor, torch.Tensor]:
    return ops.rope(q, positions), ops.rope(k, positions)
