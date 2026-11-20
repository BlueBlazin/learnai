# BUG: only the cosine table is multiplied by the table factor; the sine table is left unscaled, so rotated vectors change length with position.
import math

import torch


def yarn_inv_freq(
    dim: int,
    base: float,
    factor: float,
    original_max_positions: int,
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
    truncate: bool = True,
) -> torch.Tensor:
    def boundary(rotations: float) -> float:
        # The (fractional) pair index whose rotation completes `rotations` full turns
        # over the original context: solve L0 * base^(-2i / dim) = rotations * 2 pi for i.
        turns = original_max_positions / (rotations * 2 * math.pi)
        return dim * math.log(turns) / (2 * math.log(base))

    low, high = boundary(beta_fast), boundary(beta_slow)
    if truncate:
        low, high = math.floor(low), math.ceil(high)
    low, high = max(low, 0), min(high, dim - 1)
    # 0 for the fast pairs (keep), 1 for the slow pairs (divide by factor), linear between.
    index = torch.arange(dim // 2, dtype=torch.float32)
    ramp = torch.clamp((index - low) / (high - low), 0.0, 1.0)
    original = base ** (-torch.arange(0, dim, 2, dtype=torch.float32) / dim)
    return (1.0 - ramp) * original + ramp * (original / factor)


def yarn_tables(
    positions: torch.Tensor, inv_freq: torch.Tensor, factor: float
) -> tuple[torch.Tensor, torch.Tensor]:
    angles = positions.to(torch.float32)[:, None] * inv_freq  # (T, dim/2)
    # Both tables are scaled, so every rotated query and key grows by this factor and
    # every attention logit by its square.
    scale = 0.1 * math.log(factor) + 1.0
    return torch.cos(angles) * scale, torch.sin(angles)
