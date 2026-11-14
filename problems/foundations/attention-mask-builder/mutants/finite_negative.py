# BUG: fills masked entries with -1e9 instead of -inf; the statement requires -inf, and a large finite number stops being a mask once two masks are added or the scores are in half precision.
import torch


def build_mask(lengths: torch.Tensor, T: int, padding_side: str) -> torch.Tensor:
    pos = torch.arange(T, device=lengths.device)
    if padding_side == "right":
        real = pos[None, :] < lengths[:, None]
    elif padding_side == "left":
        real = pos[None, :] >= T - lengths[:, None]
    else:
        raise ValueError(f"padding_side must be 'left' or 'right', got {padding_side!r}")
    causal = pos[None, :] <= pos[:, None]
    allowed = (causal[None, :, :] & real[:, None, :]) | (pos[None, :] == pos[:, None])
    mask = torch.zeros(allowed.shape, dtype=torch.float32, device=lengths.device)
    mask = mask.masked_fill(~allowed, -1e9)
    return mask[:, None, :, :]
