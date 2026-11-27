# BUG: patchify reshapes straight to the target shape with no transpose, so each "patch" is p * p consecutive pixels of the row-major image; right only when W == p or p == 1.
import torch


def group_columns(w: torch.Tensor, g: int) -> torch.Tensor:
    rows, cols = w.shape
    return w.reshape(rows, cols // g, g)


def interleave(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    pairs = torch.stack((a, b), dim=-1)
    return pairs.reshape(*a.shape[:-1], 2 * a.shape[-1])


def patchify(x: torch.Tensor, p: int) -> torch.Tensor:
    batch, height, width = x.shape
    return x.reshape(batch, (height // p) * (width // p), p * p)
