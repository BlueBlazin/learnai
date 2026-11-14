import jax


def swiglu(
    x: jax.Array,
    w_gate: jax.Array,
    w_up: jax.Array,
    w_down: jax.Array,
) -> jax.Array:
    # Weights are (out, in), so a projection is x @ W^T: (..., d) -> (..., h).
    gate = x @ w_gate.T
    up = x @ w_up.T
    # Only the gate branch goes through SiLU; the up branch stays linear.
    hidden = jax.nn.silu(gate) * up
    return hidden @ w_down.T  # (..., h) -> (..., d_out)
