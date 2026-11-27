# BUG: length_mask builds the left-padded mask as the complement of the right-padded one, pos >= lengths: row b gets T - lengths[b] true entries instead of lengths[b].
import torch


def length_mask(lengths: torch.Tensor, T: int, side: str) -> torch.Tensor:
    pos = torch.arange(T)[None, :]
    n = lengths[:, None]
    if side == "right":
        return pos < n
    return pos >= n


def band_mask(T: int, w: int) -> torch.Tensor:
    i = torch.arange(T)[:, None]  # (T, 1): the row index
    j = torch.arange(T)[None, :]  # (1, T): the column index
    back = i - j  # (T, T): how far column j is behind row i
    return (back >= 0) & (back < w)


def same_segment(seg: torch.Tensor) -> torch.Tensor:
    return seg[:, None] == seg[None, :]
