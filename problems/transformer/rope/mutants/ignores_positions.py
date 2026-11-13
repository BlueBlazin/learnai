# BUG: ignores the positions argument and numbers the tokens 0 .. T - 1, which is wrong as soon as the call does not start at position 0.
import torch


def apply_rope(x: torch.Tensor, positions: torch.Tensor, base: float = 10000.0) -> torch.Tensor:
    dim = x.shape[-1]
    theta = base ** (-torch.arange(0, dim, 2, dtype=torch.float32) / dim)
    angles = torch.arange(x.shape[-2], dtype=torch.float32)[:, None] * theta
    cos, sin = torch.cos(angles), torch.sin(angles)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    rotated = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
    return rotated.reshape(x.shape)
