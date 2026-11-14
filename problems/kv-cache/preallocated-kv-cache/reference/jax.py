import jax
import jax.numpy as jnp

from whetstone.kit import ops

Cache = tuple[jax.Array, jax.Array, int]


def cache_init(capacity: int, batch: int, heads: int, dim: int) -> Cache:
    shape = (batch, heads, capacity, dim)
    return jnp.zeros(shape), jnp.zeros(shape), 0


def cache_write(cache: Cache, k_new: jax.Array, v_new: jax.Array) -> Cache:
    k, v, length = cache
    end = length + k_new.shape[2]
    if end > k.shape[2]:
        raise ValueError(f"writing {k_new.shape[2]} tokens at length {length} exceeds {k.shape[2]}")
    # Fill slots length .. end - 1 and touch no other slot. JAX arrays are immutable, so
    # the update is spelled `.at[...].set(...)`: it returns an array of the same shape,
    # and under `jit` with a donated buffer it is compiled to a write in place.
    k = k.at[:, :, length:end].set(k_new)
    v = v.at[:, :, length:end].set(v_new)
    return k, v, end


def cache_attend(q: jax.Array, cache: Cache) -> jax.Array:
    k, v, length = cache
    new = q.shape[2]
    # Only the first `length` slots hold tokens. Slicing them off keeps whatever the
    # tail contains out of the arithmetic altogether.
    k, v = k[:, :, :length], v[:, :, :length]
    # The queries belong to the last `new` tokens: query i is at position
    # length - new + i and sees the slots up to and including its own.
    visible = jnp.arange(length)[None, :] <= (length - new) + jnp.arange(new)[:, None]
    return ops.attention(q, k, v, visible)
