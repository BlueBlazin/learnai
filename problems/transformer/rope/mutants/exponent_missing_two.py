# BUG: uses theta_i = base^(-i / D) instead of base^(-2i / D), so the frequencies only fall to about base^(-1/2) instead of to about 1 / base.
import torch


def apply_rope(x: torch.Tensor, positions: torch.Tensor, base: float = 10000.0) -> torch.Tensor:
    dim = x.shape[-1]
    theta = base ** (-torch.arange(0, dim // 2, dtype=torch.float32) / dim)
    angles = positions.to(torch.float32)[:, None] * theta
    cos, sin = torch.cos(angles), torch.sin(angles)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    rotated = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
    return rotated.reshape(x.shape)
