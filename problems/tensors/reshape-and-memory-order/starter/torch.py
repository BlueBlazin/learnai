import torch


def group_columns(w: torch.Tensor, g: int) -> torch.Tensor:
    """Cut the columns of a matrix into consecutive groups of g.

    w: float32, shape (R, C).
    g: a Python int that divides C.
    Returns float32, shape (R, C // g, g), with out[r, k, i] = w[r, k * g + i].
    """
    raise NotImplementedError


def interleave(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Alternate the entries of a and b along the last axis.

    a, b: float32, the same shape (..., n), with 1 to 3 axes.
    Returns float32, shape (..., 2 * n), with out[..., 2 * i] = a[..., i] and
    out[..., 2 * i + 1] = b[..., i].
    """
    raise NotImplementedError


def patchify(x: torch.Tensor, p: int) -> torch.Tensor:
    """Cut each image into p-by-p patches and flatten every patch.

    x: float32, shape (B, H, W).
    p: a Python int that divides H and W.
    Returns float32, shape (B, (H // p) * (W // p), p * p). Patches are in row-major
    order (left to right, then top to bottom), and each patch is flattened row-major:
    out[b, i * (W // p) + j, u * p + v] = x[b, i * p + u, j * p + v].
    """
    raise NotImplementedError
