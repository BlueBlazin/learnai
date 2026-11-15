import mlx.core as mx

from mlx import nn


def split_fused(w: mx.array, parts: list[int]) -> list[mx.array]:
    """Split a fused projection weight into the weights of its parts.

    w: float32, shape (sum(parts), d_in), in (out, in) layout.
    parts: the output width of each projection, in order (positive ints).
    Returns a list of len(parts) float32 tensors; the i-th has shape (parts[i], d_in).
    """
    raise NotImplementedError


def fuse(ws: list[mx.array]) -> mx.array:
    """Fuse projection weights that read the same input into one weight.

    ws: a non-empty list of float32 tensors of shapes (n_i, d_in), in (out, in) layout.
    Returns float32, shape (sum(n_i), d_in), so that
    split_fused(fuse(ws), [n_0, n_1, ...]) gives back ws.
    """
    raise NotImplementedError


def swiglu_fused(x: mx.array, w_in: mx.array, w_down: mx.array) -> mx.array:
    """SwiGLU whose gate and up projections are fused in one matrix, gate rows first.

    x: float32, shape (..., d).
    w_in: float32, shape (2 * h, d). Rows 0 .. h-1 are the gate projection, rows
        h .. 2h-1 the up projection.
    w_down: float32, shape (d_out, h).
    Returns float32, shape (..., d_out): (silu(gate) * up) @ w_down^T.
    """
    raise NotImplementedError
