import jax
import jax.numpy as jnp


def mse(a: jax.Array, b: jax.Array) -> jax.Array:
    """Mean squared error: the mean of (a - b)**2 over every element.

    a, b: float32 arrays of the same shape, 1 to 4 dimensions.
    Returns a 0-d float32 array.
    """
    raise NotImplementedError


def sqnr_db(ref: jax.Array, approx: jax.Array) -> jax.Array:
    """Signal-to-quantization-noise ratio in decibels: 10 log10(sum ref**2 / sum (ref - approx)**2).

    ref, approx: float32 arrays of the same shape, 1 to 4 dimensions; ref is not all zeros.
    Returns a 0-d float32 array; +inf when approx equals ref everywhere.
    """
    raise NotImplementedError


def max_abs_error(a: jax.Array, b: jax.Array) -> jax.Array:
    """Largest absolute difference: the maximum of |a - b| over every element.

    a, b: float32 arrays of the same shape, 1 to 4 dimensions.
    Returns a 0-d float32 array.
    """
    raise NotImplementedError


def output_error(w: jax.Array, w_hat: jax.Array, x: jax.Array) -> jax.Array:
    """Relative error of a linear layer's output: ||x w^T - x w_hat^T||_F / ||x w^T||_F.

    w, w_hat: float32 arrays of shape (out, in). x: float32 array of shape (n, in).
    Returns a 0-d float32 array.
    """
    raise NotImplementedError
