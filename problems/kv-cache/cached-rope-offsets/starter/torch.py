import torch

from whetstone.kit import ops


def rope_positions(cache_lengths: torch.Tensor, n_new: int) -> torch.Tensor:
    """Absolute positions of the next n_new tokens of every batch row.

    cache_lengths: integer tensor, shape (B,). Row b has cache_lengths[b] tokens cached.
    n_new: int >= 1. The number of new tokens, the same for every row.
    Returns an integer tensor of shape (B, n_new).
    """
    raise NotImplementedError


def rotate_new(
    q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor, theta: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to the queries and keys of the new tokens.

    q, k: float32, shape (B, H, T, D) with D even. New queries and keys, not yet rotated.
    positions: integer tensor, shape (B, T). The absolute position of each new token.
    theta: float > 1. The RoPE base: pair i turns by position * theta ** (-2 * i / D).
    Returns (q_rot, k_rot), both float32 of shape (B, H, T, D).
    """
    raise NotImplementedError
