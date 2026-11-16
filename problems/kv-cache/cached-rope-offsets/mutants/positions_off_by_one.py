# BUG: the first new token gets position cache_length + 1; positions are 0-based, so a row with n cached tokens continues at n.
import torch

from learnai.kit import ops


def rope_positions(cache_lengths: torch.Tensor, n_new: int) -> torch.Tensor:
    return cache_lengths[:, None] + 1 + torch.arange(n_new)


def rotate_new(
    q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor, theta: float
) -> tuple[torch.Tensor, torch.Tensor]:
    return ops.rope(q, positions, theta), ops.rope(k, positions, theta)
