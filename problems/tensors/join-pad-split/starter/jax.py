import jax
import jax.numpy as jnp


def pad_rows(rows: list[jax.Array], pad: int, side: str) -> tuple[jax.Array, jax.Array]:
    """Pad rows of different lengths into one batch, and mark the real tokens.

    rows: a list of B tensors, int32, row i of shape (n_i,) with n_i >= 1.
    pad: the id written at the padding positions.
    side: "left" (padding first, tokens last) or "right" (tokens first).
    Returns (ids, mask) with T_max = max n_i:
      ids: an integer tensor, shape (B, T_max);
      mask: bool, shape (B, T_max), True at the positions that hold a token of the row.
    """
    raise NotImplementedError


def shift_right(ids: jax.Array, bos: int) -> jax.Array:
    """Move every row one position to the right, with bos first.

    ids: int32, shape (B, T).
    Returns an integer tensor, shape (B, T), with out[b, 0] = bos and
    out[b, t] = ids[b, t - 1] for t >= 1. The last column of ids is dropped.
    """
    raise NotImplementedError


def stack_steps(steps: list[jax.Array]) -> jax.Array:
    """Join the per-step outputs of a decode loop along a new step axis.

    steps: a list of T tensors, float32, each of shape (B, V).
    Returns float32, shape (B, T, V), with out[b, t] = steps[t][b].
    """
    raise NotImplementedError
