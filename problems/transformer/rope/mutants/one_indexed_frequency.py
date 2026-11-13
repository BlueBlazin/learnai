# BUG: starts the frequency index at 1 (theta_i = base^(-2(i + 1) / D)), a mistranslation of the 1-based formula in the paper; the first pair should have theta = 1.
import torch


def apply_rope(x: torch.Tensor, positions: torch.Tensor, base: float = 10000.0) -> torch.Tensor:
    dim = x.shape[-1]
    theta = base ** (-torch.arange(2, dim + 2, 2, dtype=torch.float32) / dim)
    angles = positions.to(torch.float32)[:, None] * theta
    cos, sin = torch.cos(angles), torch.sin(angles)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    rotated = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
    return rotated.reshape(x.shape)
