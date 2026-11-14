import jax
import jax.numpy as jnp

from whetstone.kit import ops

Cache = tuple[jax.Array, jax.Array, int]  # (k, v, length)


def cache_init(capacity: int, batch: int, heads: int, dim: int) -> Cache:
    """An empty cache with room for `capacity` tokens.

    Returns (k, v, 0): k and v are float32 zeros of shape (batch, heads, capacity, dim).
    """
    raise NotImplementedError


def cache_write(cache: Cache, k_new: jax.Array, v_new: jax.Array) -> Cache:
    """Store the keys and values of T new tokens in slots length .. length + T - 1.

    cache: (k, v, length) with k, v float32 of shape (B, H, C, D) and 0 <= length <= C.
    k_new, v_new: float32, shape (B, H, T, D).
    Returns (k, v, length + T) with k and v still of shape (B, H, C, D).
    Raises ValueError if length + T > C.
    """
    raise NotImplementedError


def cache_attend(q: jax.Array, cache: Cache) -> jax.Array:
    """Attend from the queries of the T most recently written tokens over the cache.

    q: float32, shape (B, H, T, D), 1 <= T <= length. Query i belongs to the token in
        slot length - T + i and attends to slots 0 .. length - T + i.
    cache: (k, v, length). Slots at index >= length hold arbitrary data (possibly NaN).
    Returns float32, shape (B, H, T, D).
    """
    raise NotImplementedError
