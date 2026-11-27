import jax
import jax.numpy as jnp


def pick_pairs(x: jax.Array, rows: jax.Array, cols: jax.Array) -> jax.Array:
    """One entry per (row, column) pair.

    x: float32, shape (H, W).
    rows, cols: integer, shape (N,), with 0 <= rows[n] < H and 0 <= cols[n] < W.
    Returns float32, shape (N,), with out[n] = x[rows[n], cols[n]].
    """
    raise NotImplementedError


def pick_block(x: jax.Array, rows: jax.Array, cols: jax.Array) -> jax.Array:
    """The block where the chosen rows cross the chosen columns.

    x: float32, shape (H, W).
    rows: integer, shape (R,), with 0 <= rows[r] < H.
    cols: integer, shape (C,), with 0 <= cols[c] < W.
    Returns float32, shape (R, C), with out[r, c] = x[rows[r], cols[c]].
    """
    raise NotImplementedError


def pick_per_row(x: jax.Array, cols: jax.Array) -> jax.Array:
    """K steps from every row of a batch, each row with its own choice.

    x: float32, shape (B, T, D).
    cols: integer, shape (B, K), with 0 <= cols[b, k] < T.
    Returns float32, shape (B, K, D), with out[b, k] = x[b, cols[b, k]].
    """
    raise NotImplementedError
