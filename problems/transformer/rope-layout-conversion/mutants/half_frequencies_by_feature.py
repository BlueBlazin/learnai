# BUG: rope_half builds its angle table as for the interleaved layout (each frequency repeated for two adjacent features) and reads it by feature index, so pair i turns at the frequency of pair i // 2 in the first half and of pair (i + D/2) // 2 in the second.
import torch


def _cos_sin(dim: int, pos: torch.Tensor, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    """cos and sin of pos[t] * theta^(-2i / dim), shape (T, dim / 2). Both layouts use it."""
    freqs = theta ** (-torch.arange(0, dim, 2, dtype=torch.float32) / dim)
    angles = pos.to(torch.float32)[:, None] * freqs
    return torch.cos(angles), torch.sin(angles)


def rope_interleaved(x: torch.Tensor, pos: torch.Tensor, theta: float = 10000.0) -> torch.Tensor:
    cos, sin = _cos_sin(x.shape[-1], pos, theta)
    # Pair i is features (2i, 2i + 1).
    a, b = x[..., 0::2], x[..., 1::2]
    rotated = torch.stack((a * cos - b * sin, a * sin + b * cos), dim=-1)
    return rotated.reshape(x.shape)  # (..., D/2, 2) -> (..., D): pairs back side by side


def rope_half(x: torch.Tensor, pos: torch.Tensor, theta: float = 10000.0) -> torch.Tensor:
    cos, sin = _cos_sin(x.shape[-1], pos, theta)
    # Pair i is features (i, i + D/2): first members in the first half, second in the second.
    half = x.shape[-1] // 2
    cos, sin = cos.repeat_interleave(2, dim=-1), sin.repeat_interleave(2, dim=-1)  # (T, D)
    rotated = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
    return x * cos + rotated * sin


def permute_qk(w: torch.Tensor, d_head: int, n_heads: int) -> torch.Tensor:
    # Rows are output features. Within each head, row 2i + c (pair i, member c) of the
    # interleaved layout moves to row c * d_head/2 + i of the half-split layout.
    pairs = w.reshape(n_heads, d_head // 2, 2, w.shape[-1])  # [head, pair, member, in]
    return pairs.transpose(1, 2).reshape(w.shape)  # [head, member, pair, in], flattened
