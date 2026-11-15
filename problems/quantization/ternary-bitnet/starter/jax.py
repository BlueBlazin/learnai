import jax
import jax.numpy as jnp


def ternary_quantize(w: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Quantize w to {-1, 0, 1} with one scale, the mean absolute value of w.

    w: float32 array of shape (out, in).
    Returns (t, scale): an int8 array of shape (out, in) with values -1, 0, or 1, and
    a 0-d float32 array (1 if every weight is 0). round() rounds half to even.
    """
    raise NotImplementedError


def ternary_matmul(x: jax.Array, t: jax.Array, scale: jax.Array) -> jax.Array:
    """x @ (scale * t).T, computed by adding and subtracting selected inputs.

    x: float32 array of shape (..., in). t: int8 array of shape (out, in) with values
    -1, 0, or 1. scale: 0-d float32 array.
    Returns a float32 array of shape (..., out).
    """
    raise NotImplementedError
