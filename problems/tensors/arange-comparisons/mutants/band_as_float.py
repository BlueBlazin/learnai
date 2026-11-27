# BUG: band_mask subtracts two float triangles and returns the band as float32 zeros and ones instead of a boolean tensor.
import torch


def length_mask(lengths: torch.Tensor, T: int, side: str) -> torch.Tensor:
    pos = torch.arange(T)[None, :]  # (1, T): the position j of every column
    n = lengths[:, None]  # (B, 1): the length of every row
    if side == "right":
        return pos < n  # the first n positions are real
    return pos >= T - n  # the last n positions are real


def band_mask(T: int, w: int) -> torch.Tensor:
    ones = torch.ones(T, T)
    return torch.tril(ones) - torch.tril(ones, diagonal=-w)


def same_segment(seg: torch.Tensor) -> torch.Tensor:
    return seg[:, None] == seg[None, :]
