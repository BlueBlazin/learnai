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
    # A (T, 1) column against a (1, T) row: every pair of positions.
    return seg[:, None] == seg[None, :]
