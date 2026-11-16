# BUG: only the queries are rotated; the new keys are returned as they came, on the idea that keys in a cache are already rotated (the cached ones are, the new ones are not yet).
import torch

from learnai.kit import ops


def rope_positions(cache_lengths: torch.Tensor, n_new: int) -> torch.Tensor:
    return cache_lengths[:, None] + torch.arange(n_new)


def rotate_new(
    q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor, theta: float
) -> tuple[torch.Tensor, torch.Tensor]:
    return ops.rope(q, positions, theta), k
