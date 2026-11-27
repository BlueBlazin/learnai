# BUG: interleave stacks a and b on a new first axis and reshapes, so memory order is all of a and then all of b, redrawn into rows of 2n; nothing alternates.
import torch


def group_columns(w: torch.Tensor, g: int) -> torch.Tensor:
    rows, cols = w.shape
    return w.reshape(rows, cols // g, g)


def interleave(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    pairs = torch.stack((a, b), dim=0)
    return pairs.reshape(*a.shape[:-1], 2 * a.shape[-1])


def patchify(x: torch.Tensor, p: int) -> torch.Tensor:
    batch, height, width = x.shape
    rows, cols = height // p, width // p
    x = x.reshape(batch, rows, p, cols, p)
    x = x.permute(0, 1, 3, 2, 4)
    return x.reshape(batch, rows * cols, p * p)
