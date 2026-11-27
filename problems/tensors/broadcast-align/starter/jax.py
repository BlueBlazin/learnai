import jax
import jax.numpy as jnp


def relative_positions(T: int) -> jax.Array:
    """The table of position differences.

    T: a Python int, T >= 1.
    Returns an integer tensor of shape (T, T) with out[i, j] = i - j.
    """
    raise NotImplementedError


def scale_rows(x: jax.Array, s: jax.Array) -> jax.Array:
    """Multiply each row of the batch by its own factor.

    x: float32, shape (B, T, D).
    s: float32, shape (B,).
    Returns float32, shape (B, T, D), with out[b, t, d] = x[b, t, d] * s[b].
    """
    raise NotImplementedError


def pairwise_sq_dist(a: jax.Array, b: jax.Array) -> jax.Array:
    """Squared distances between every point of a and every point of b.

    a: float32, shape (N, D).
    b: float32, shape (M, D).
    Returns float32, shape (N, M), with out[n, m] = sum over d of (a[n, d] - b[m, d]) ** 2.
    """
    raise NotImplementedError
