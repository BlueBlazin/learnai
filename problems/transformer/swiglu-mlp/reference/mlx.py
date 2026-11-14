import mlx.core as mx
from mlx import nn


def swiglu(
    x: mx.array,
    w_gate: mx.array,
    w_up: mx.array,
    w_down: mx.array,
) -> mx.array:
    # Weights are (out, in), so a projection is x @ W^T: (..., d) -> (..., h).
    gate = x @ w_gate.T
    up = x @ w_up.T
    # Only the gate branch goes through SiLU; the up branch stays linear.
    hidden = nn.silu(gate) * up
    return hidden @ w_down.T  # (..., h) -> (..., d_out)
