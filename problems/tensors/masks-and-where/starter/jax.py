import jax
import jax.numpy as jnp


def fill_masked(scores: jax.Array, keep: jax.Array) -> jax.Array:
    """scores, with -inf at every entry that is not kept.

    scores: float32, shape (B, V). An entry that is not kept may be inf, -inf, or NaN.
    keep: bool, shape (B, V).
    Returns float32, shape (B, V): scores[b, v] where keep[b, v], and -inf elsewhere.
    """
    raise NotImplementedError


def masked_mean(x: jax.Array, mask: jax.Array) -> jax.Array:
    """The mean of the kept steps of every row.

    x: float32, shape (B, T, D). A step that is not kept may hold inf, -inf, or NaN.
    mask: bool, shape (B, T); mask[b, t] says whether step t of row b is kept.
    Returns float32, shape (B, D): the sum of the kept steps of row b divided by their
    number. A row that keeps no step returns zeros.
    """
    raise NotImplementedError


def masked_max(x: jax.Array, mask: jax.Array) -> jax.Array:
    """The largest kept value of every feature of every row.

    x: float32, shape (B, T, D). A step that is not kept may hold inf, -inf, or NaN.
    mask: bool, shape (B, T); mask[b, t] says whether step t of row b is kept.
    Returns float32, shape (B, D): the max over the kept steps of row b. A row that
    keeps no step returns -inf in every feature.
    """
    raise NotImplementedError
