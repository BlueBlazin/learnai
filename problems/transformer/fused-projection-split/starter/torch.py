import torch
import torch.nn.functional as F


def split_fused(w: torch.Tensor, parts: list[int]) -> list[torch.Tensor]:
    """Split a fused projection weight into the weights of its parts.

    w: float32, shape (sum(parts), d_in), in (out, in) layout.
    parts: the output width of each projection, in order (positive ints).
    Returns a list of len(parts) float32 tensors; the i-th has shape (parts[i], d_in).
    """
    raise NotImplementedError


def fuse(ws: list[torch.Tensor]) -> torch.Tensor:
    """Fuse projection weights that read the same input into one weight.

    ws: a non-empty list of float32 tensors of shapes (n_i, d_in), in (out, in) layout.
    Returns float32, shape (sum(n_i), d_in), so that
    split_fused(fuse(ws), [n_0, n_1, ...]) gives back ws.
    """
    raise NotImplementedError


def swiglu_fused(x: torch.Tensor, w_in: torch.Tensor, w_down: torch.Tensor) -> torch.Tensor:
    """SwiGLU whose gate and up projections are fused in one matrix, gate rows first.

    x: float32, shape (..., d).
    w_in: float32, shape (2 * h, d). Rows 0 .. h-1 are the gate projection, rows
        h .. 2h-1 the up projection.
    w_down: float32, shape (d_out, h).
    Returns float32, shape (..., d_out): (silu(gate) * up) @ w_down^T.
    """
    raise NotImplementedError
