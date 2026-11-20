# BUG: proportional_rope rounds factor * D / 2 to the nearest integer (halves up) instead of flooring it, so it rotates one pair too many when the product has a fractional part of one half.
import torch


def partial_rope(
    x: torch.Tensor, positions: torch.Tensor, base: float, factor: float
) -> torch.Tensor:
    dim = x.shape[-1]
    r = int(dim * factor)  # rotated channels: the first r
    half = r // 2
    # The rotated block is a complete RoPE of width r: its own frequency ladder.
    freqs = base ** (-torch.arange(0, r, 2, dtype=torch.float32) / r)
    angles = positions.to(torch.float32)[:, None] * freqs  # (T, r/2)
    cos, sin = torch.cos(angles), torch.sin(angles)
    # Half-split inside the block: pair i is channels (i, i + r/2).
    a, b = x[..., :half], x[..., half:r]
    return torch.cat((a * cos - b * sin, a * sin + b * cos, x[..., r:]), dim=-1)


def proportional_rope(
    x: torch.Tensor, positions: torch.Tensor, base: float, factor: float
) -> torch.Tensor:
    dim = x.shape[-1]
    half = dim // 2
    n_rot = int(factor * dim / 2 + 0.5)
    # The first n_rot rungs of the full head's ladder; every other pair has frequency 0.
    freqs = base ** (-torch.arange(0, 2 * n_rot, 2, dtype=torch.float32) / dim)
    freqs = torch.cat((freqs, torch.zeros(half - n_rot)))
    angles = positions.to(torch.float32)[:, None] * freqs  # (T, D/2)
    cos, sin = torch.cos(angles), torch.sin(angles)
    # Half-split over the whole head: pair i is channels (i, i + D/2).
    a, b = x[..., :half], x[..., half:]
    return torch.cat((a * cos - b * sin, a * sin + b * cos), dim=-1)
