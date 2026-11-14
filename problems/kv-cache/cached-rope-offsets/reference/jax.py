import jax
import jax.numpy as jnp

from whetstone.kit import ops


def rope_positions(cache_lengths: jax.Array, n_new: int) -> jax.Array:
    # Row b already holds cache_lengths[b] tokens, at positions 0 .. cache_lengths[b] - 1,
    # so its new tokens continue at cache_lengths[b], cache_lengths[b] + 1, ...
    return cache_lengths[:, None] + jnp.arange(n_new)  # (B, 1) + (n_new,) -> (B, n_new)


def rotate_new(
    q: jax.Array, k: jax.Array, positions: jax.Array, theta: float
) -> tuple[jax.Array, jax.Array]:
    # Only the new tokens are rotated, each for its own absolute position. The cached
    # keys were rotated when they were new and are never touched again.
    return ops.rope(q, positions, theta), ops.rope(k, positions, theta)
