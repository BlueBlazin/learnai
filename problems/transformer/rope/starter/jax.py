import jax
import jax.numpy as jnp


def apply_rope(x: jax.Array, positions: jax.Array, base: float = 10000.0) -> jax.Array:
    """Rotate consecutive pairs of x by angles that grow with the token position.

    x: float32, shape (B, H, T, D) with D even. Queries or keys, one vector per head and token.
    positions: integer array, shape (T,). The absolute position of each of the T tokens.
    base: float > 1. Pair i rotates by positions[t] * base ** (-2 * i / D) radians.
    Returns float32, shape (B, H, T, D).
    """
    raise NotImplementedError
