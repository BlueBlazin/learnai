# BUG: same_segment assumes a segment is one contiguous run: it numbers the runs with a running count of changes and compares those, so two separate runs with the same id are treated as different segments.
import torch


def length_mask(lengths: torch.Tensor, T: int, side: str) -> torch.Tensor:
    pos = torch.arange(T)[None, :]  # (1, T): the position j of every column
    n = lengths[:, None]  # (B, 1): the length of every row
    if side == "right":
        return pos < n  # the first n positions are real
    return pos >= T - n  # the last n positions are real


def band_mask(T: int, w: int) -> torch.Tensor:
    i = torch.arange(T)[:, None]  # (T, 1): the row index
    j = torch.arange(T)[None, :]  # (1, T): the column index
    back = i - j  # (T, T): how far column j is behind row i
    return (back >= 0) & (back < w)


def same_segment(seg: torch.Tensor) -> torch.Tensor:
    change = torch.ones_like(seg)
    change[1:] = (seg[1:] != seg[:-1]).to(seg.dtype)
    run = torch.cumsum(change, dim=0)
    return run[:, None] == run[None, :]
