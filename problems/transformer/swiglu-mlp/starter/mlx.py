import mlx.core as mx
from mlx import nn


def swiglu(
    x: mx.array,
    w_gate: mx.array,
    w_up: mx.array,
    w_down: mx.array,
) -> mx.array:
    """SwiGLU feed-forward layer: (silu(x W_gate^T) * (x W_up^T)) W_down^T.

    x: float32, shape (..., d). One to three dimensions; the last axis is the feature axis.
    w_gate, w_up: float32, shape (h, d). Gate and up projections, (out, in) layout.
    w_down: float32, shape (d_out, h). Down projection, (out, in) layout.
    Returns float32, shape (..., d_out).
    """
    raise NotImplementedError
