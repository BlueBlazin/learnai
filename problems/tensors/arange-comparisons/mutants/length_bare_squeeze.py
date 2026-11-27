# BUG: length_mask ends in a bare squeeze(), which removes the batch axis of a single row and the position axis when T == 1.
import torch


def length_mask(lengths: torch.Tensor, T: int, side: str) -> torch.Tensor:
    pos = torch.arange(T)[None, :]
    n = lengths[:, None]
    if side == "right":
        return (pos < n).squeeze()
    return (pos >= T - n).squeeze()


def band_mask(T: int, w: int) -> torch.Tensor:
    i = torch.arange(T)[:, None]  # (T, 1): the row index
    j = torch.arange(T)[None, :]  # (1, T): the column index
    back = i - j  # (T, T): how far column j is behind row i
    return (back >= 0) & (back < w)


def same_segment(seg: torch.Tensor) -> torch.Tensor:
    return seg[:, None] == seg[None, :]
