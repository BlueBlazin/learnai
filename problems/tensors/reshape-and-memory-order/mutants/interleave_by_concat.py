# BUG: interleave concatenates a and b along the last axis: all of a, then all of b, instead of alternating entries; right only when n == 1.
import torch


def group_columns(w: torch.Tensor, g: int) -> torch.Tensor:
    rows, cols = w.shape
    return w.reshape(rows, cols // g, g)


def interleave(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.cat((a, b), dim=-1)


def patchify(x: torch.Tensor, p: int) -> torch.Tensor:
    batch, height, width = x.shape
    rows, cols = height // p, width // p
    x = x.reshape(batch, rows, p, cols, p)
    x = x.permute(0, 1, 3, 2, 4)
    return x.reshape(batch, rows * cols, p * p)
