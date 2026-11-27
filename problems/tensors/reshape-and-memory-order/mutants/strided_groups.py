# BUG: group_columns reshapes to (R, g, C // g) and swaps the last two axes, so group k holds every (C // g)-th column starting at k instead of g consecutive columns.
import torch


def group_columns(w: torch.Tensor, g: int) -> torch.Tensor:
    rows, cols = w.shape
    return w.reshape(rows, g, cols // g).transpose(1, 2)


def interleave(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    pairs = torch.stack((a, b), dim=-1)
    return pairs.reshape(*a.shape[:-1], 2 * a.shape[-1])


def patchify(x: torch.Tensor, p: int) -> torch.Tensor:
    batch, height, width = x.shape
    rows, cols = height // p, width // p
    x = x.reshape(batch, rows, p, cols, p)
    x = x.permute(0, 1, 3, 2, 4)
    return x.reshape(batch, rows * cols, p * p)
