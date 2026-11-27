import jax
import jax.numpy as jnp


def repeat_kv_heads(k: jax.Array, n_rep: int) -> jax.Array:
    """Give every key/value group n_rep consecutive heads.

    k: float32, shape (B, G, T, D).
    n_rep: int, at least 1.
    Returns float32, shape (B, G * n_rep, T, D), with
    out[b, g * n_rep + r] = k[b, g] for 0 <= r < n_rep.
    """
    raise NotImplementedError


def expand_for_beams(x: jax.Array, K: int) -> jax.Array:
    """Give every row K consecutive copies.

    x: int32, shape (B, T).
    K: int, at least 1.
    Returns an integer tensor, shape (B * K, T), with out[b * K + j] = x[b] for
    0 <= j < K.
    """
    raise NotImplementedError


def tile_positions(pos: jax.Array, B: int) -> jax.Array:
    """The same row of positions for every batch row.

    pos: int32, shape (T,).
    B: int, at least 1.
    Returns an integer tensor, shape (B, T), with out[b, t] = pos[t].
    """
    raise NotImplementedError
