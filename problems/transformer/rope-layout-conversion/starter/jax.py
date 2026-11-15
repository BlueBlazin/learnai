import jax
import jax.numpy as jnp


def rope_interleaved(x: jax.Array, pos: jax.Array, theta: float = 10000.0) -> jax.Array:
    """RoPE in the interleaved layout: pair i is features (2i, 2i + 1).

    x: float32, shape (B, H, T, D) with D even. Queries or keys.
    pos: integer tensor, shape (T,). The absolute position of each token.
    theta: float > 1. Pair i turns by pos[t] * theta ** (-2 * i / D) radians.
    Returns float32, shape (B, H, T, D).
    """
    raise NotImplementedError


def rope_half(x: jax.Array, pos: jax.Array, theta: float = 10000.0) -> jax.Array:
    """RoPE in the half-split layout: pair i is features (i, i + D/2).

    Same arguments, frequencies, and direction of rotation as rope_interleaved.
    Returns float32, shape (B, H, T, D).
    """
    raise NotImplementedError


def permute_qk(w: jax.Array, d_head: int, n_heads: int) -> jax.Array:
    """Convert a query or key projection from the interleaved to the half-split layout.

    w: float32, shape (n_heads * d_head, d_in), in (out, in) layout; head h owns rows
        h * d_head to (h + 1) * d_head - 1. Trained for rope_interleaved.
    Returns float32 of the same shape: the weight whose projections, rotated with
    rope_half, give the same attention scores as w's rotated with rope_interleaved.
    """
    raise NotImplementedError
