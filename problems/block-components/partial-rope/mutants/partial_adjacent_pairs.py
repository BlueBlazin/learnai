# BUG: partial_rope rotates adjacent pairs (0, 1), (2, 3), ... inside the first r channels (the interleaved layout) instead of the half-split pairs (i, i + r/2).
import torch


def partial_rope(
    x: torch.Tensor, positions: torch.Tensor, base: float, factor: float
) -> torch.Tensor:
    dim = x.shape[-1]
    r = int(dim * factor)  # rotated channels: the first r
    # The rotated block is a complete RoPE of width r: its own frequency ladder.
    freqs = base ** (-torch.arange(0, r, 2, dtype=torch.float32) / r)
    angles = positions.to(torch.float32)[:, None] * freqs  # (T, r/2)
    cos, sin = torch.cos(angles), torch.sin(angles)
    a, b = x[..., 0:r:2], x[..., 1:r:2]
    rotated = torch.stack((a * cos - b * sin, a * sin + b * cos), dim=-1)
    return torch.cat((rotated.reshape(*x.shape[:-1], r), x[..., r:]), dim=-1)


def proportional_rope(
    x: torch.Tensor, positions: torch.Tensor, base: float, factor: float
) -> torch.Tensor:
    dim = x.shape[-1]
    half = dim // 2
    n_rot = int(factor * dim // 2)  # rotated pairs: the first n_rot of dim/2
    # The first n_rot rungs of the full head's ladder; every other pair has frequency 0.
    freqs = base ** (-torch.arange(0, 2 * n_rot, 2, dtype=torch.float32) / dim)
    freqs = torch.cat((freqs, torch.zeros(half - n_rot)))
    angles = positions.to(torch.float32)[:, None] * freqs  # (T, D/2)
    cos, sin = torch.cos(angles), torch.sin(angles)
    # Half-split over the whole head: pair i is channels (i, i + D/2).
    a, b = x[..., :half], x[..., half:]
    return torch.cat((a * cos - b * sin, a * sin + b * cos), dim=-1)
