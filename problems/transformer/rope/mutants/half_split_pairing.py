# BUG: pairs dimension i with dimension i + D/2 (the GPT-NeoX layout) instead of the adjacent dimensions 2i and 2i + 1.
import torch


def apply_rope(x: torch.Tensor, positions: torch.Tensor, base: float = 10000.0) -> torch.Tensor:
    dim = x.shape[-1]
    theta = base ** (-torch.arange(0, dim, 2, dtype=torch.float32) / dim)
    angles = positions.to(torch.float32)[:, None] * theta
    cos, sin = torch.cos(angles), torch.sin(angles)
    x1, x2 = x[..., : dim // 2], x[..., dim // 2 :]
    return torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
