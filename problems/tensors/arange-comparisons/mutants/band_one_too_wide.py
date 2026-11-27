# BUG: band_mask compares with <= w: the window holds w + 1 positions.
import torch


def length_mask(lengths: torch.Tensor, T: int, side: str) -> torch.Tensor:
    pos = torch.arange(T)[None, :]  # (1, T): the position j of every column
    n = lengths[:, None]  # (B, 1): the length of every row
    if side == "right":
        return pos < n  # the first n positions are real
    return pos >= T - n  # the last n positions are real


def band_mask(T: int, w: int) -> torch.Tensor:
    i = torch.arange(T)[:, None]
    j = torch.arange(T)[None, :]
    back = i - j
    return (back >= 0) & (back <= w)


def same_segment(seg: torch.Tensor) -> torch.Tensor:
    return seg[:, None] == seg[None, :]
