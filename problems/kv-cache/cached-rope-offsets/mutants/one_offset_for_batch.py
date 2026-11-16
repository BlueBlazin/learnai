# BUG: one offset for the whole batch: every row continues from the first row's cache length.
import torch

from learnai.kit import ops


def rope_positions(cache_lengths: torch.Tensor, n_new: int) -> torch.Tensor:
    return cache_lengths[0] + torch.arange(n_new).expand(cache_lengths.shape[0], n_new)


def rotate_new(
    q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor, theta: float
) -> tuple[torch.Tensor, torch.Tensor]:
    return ops.rope(q, positions, theta), ops.rope(k, positions, theta)
