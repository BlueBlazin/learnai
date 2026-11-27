# BUG: patchify flattens each patch column-major (pixel (u, v) lands at v * p + u) by putting the in-patch column axis before the in-patch row axis.
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
    x = x.permute(0, 1, 3, 4, 2)
    return x.reshape(batch, rows * cols, p * p)
