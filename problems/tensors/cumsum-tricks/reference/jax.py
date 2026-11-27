import jax
import jax.numpy as jnp


def exclusive_cumsum(x: jax.Array) -> jax.Array:
    # The inclusive sums, moved one place to the right: position i gets the sum that
    # ended at i - 1, position 0 gets 0, and the last inclusive sum is dropped.
    total = jnp.cumsum(x, axis=-1)
    zero = jnp.zeros_like(x[..., :1])  # a slice, so the axis is kept with size 1
    return jnp.concatenate([zero, total[..., :-1]], axis=-1)


def first_true(mask: jax.Array) -> jax.Array:
    # The running count of true entries is 0 exactly at the positions before the first
    # one. There are as many of those as the index of the first true entry, and T of
    # them when the row has none.
    seen = jnp.cumsum(mask, axis=1)  # (B, T) int32
    return (seen == 0).sum(axis=1)


def through_first_eos(ids: jax.Array, eos: int) -> jax.Array:
    # Position t is kept when no eos comes strictly before it. The exclusive count
    # leaves the first eos itself at 0, so it is kept, and everything after it is not.
    is_eos = (ids == eos).astype(jnp.int32)
    return exclusive_cumsum(is_eos) == 0
