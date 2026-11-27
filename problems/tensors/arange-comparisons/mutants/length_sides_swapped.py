# BUG: length_mask has its sides swapped: "right" marks the last lengths[b] positions and "left" the first.
import torch


def length_mask(lengths: torch.Tensor, T: int, side: str) -> torch.Tensor:
    pos = torch.arange(T)[None, :]
    n = lengths[:, None]
    if side == "left":
        return pos < n
    return pos >= T - n


def band_mask(T: int, w: int) -> torch.Tensor:
    i = torch.arange(T)[:, None]  # (T, 1): the row index
    j = torch.arange(T)[None, :]  # (1, T): the column index
    back = i - j  # (T, T): how far column j is behind row i
    return (back >= 0) & (back < w)


def same_segment(seg: torch.Tensor) -> torch.Tensor:
    return seg[:, None] == seg[None, :]
