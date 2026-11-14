import jax
import jax.numpy as jnp

from whetstone.kit import ops


def attend_step(
    q_new: jax.Array,
    k_new: jax.Array,
    v_new: jax.Array,
    cache: tuple[jax.Array, jax.Array] | None,
) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
    """One attention step over a KV cache: append the new keys and values, then attend.

    q_new, k_new, v_new: float32, shape (B, H, T, D). Queries, keys, and values of the
        T new tokens, already projected and split into heads.
    cache: None (nothing cached yet), or a tuple (k, v) of float32 tensors of shape
        (B, H, S, D) with the keys and values of the S earlier tokens, oldest first.
    Returns (out, (k, v)): out is float32 of shape (B, H, T, D), where new token i
    attends to every cached token and to new tokens 0..i; k and v are float32 of shape
    (B, H, S + T, D), the cached entries followed by the new ones.
    """
    raise NotImplementedError
