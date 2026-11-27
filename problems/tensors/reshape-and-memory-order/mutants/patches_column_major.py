# BUG: patchify orders the patches column-major (top to bottom, then left to right) by putting the patch-column axis before the patch-row axis.
import torch


def group_columns(w: torch.Tensor, g: int) -> torch.Tensor:
    rows, cols = w.shape
    return w.reshape(rows, cols // g, g)


def interleave(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    pairs = torch.stack((a, b), dim=-1)
    return pairs.reshape(*a.shape[:-1], 2 * a.shape[-1])


def patchify(x: torch.Tensor, p: int) -> torch.Tensor:
    batch, height, width = x.shape
    rows, cols = height // p, width // p
    x = x.reshape(batch, rows, p, cols, p)
    x = x.permute(0, 3, 1, 2, 4)
    return x.reshape(batch, rows * cols, p * p)
