import jax
import jax.numpy as jnp


def token_scores(logits: jax.Array, ids: jax.Array) -> jax.Array:
    """The score of one chosen token at every position.

    logits: float32, shape (B, T, V).
    ids: int32, shape (B, T), with 0 <= ids[b, t] < V.
    Returns float32, shape (B, T), with out[b, t] = logits[b, t, ids[b, t]].
    """
    raise NotImplementedError


def take_columns(values: jax.Array, idx: jax.Array) -> jax.Array:
    """K entries from every row, each row with its own choice.

    values: float32, shape (B, V).
    idx: int32, shape (B, K), with 0 <= idx[b, k] < V.
    Returns float32, shape (B, K), with out[b, k] = values[b, idx[b, k]].
    """
    raise NotImplementedError


def reorder_steps(x: jax.Array, perm: jax.Array) -> jax.Array:
    """Reorder the steps of every row by that row's own permutation.

    x: float32, shape (B, T, D).
    perm: int32, shape (B, T); every row is a permutation of 0 .. T - 1.
    Returns float32, shape (B, T, D), with out[b, t] = x[b, perm[b, t]].
    """
    raise NotImplementedError
