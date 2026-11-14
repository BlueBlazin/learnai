# BUG: positions restart at 0 in every call (arange(n_new) for each row), as if the new tokens were the start of the sequence.
import torch

from whetstone.kit import ops


def rope_positions(cache_lengths: torch.Tensor, n_new: int) -> torch.Tensor:
    return torch.zeros_like(cache_lengths)[:, None] + torch.arange(n_new)


def rotate_new(
    q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor, theta: float
) -> tuple[torch.Tensor, torch.Tensor]:
    return ops.rope(q, positions, theta), ops.rope(k, positions, theta)
