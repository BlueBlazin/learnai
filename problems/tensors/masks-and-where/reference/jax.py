import jax
import jax.numpy as jnp


def fill_masked(scores: jax.Array, keep: jax.Array) -> jax.Array:
    # where selects: the entry that is not kept is never used in any arithmetic, so it
    # does not matter what it holds.
    return jnp.where(keep, scores, -jnp.inf)


def masked_mean(x: jax.Array, mask: jax.Array) -> jax.Array:
    # (B, T) -> (B, T, 1), so the mask lines up with the steps and broadcasts over the
    # features. A dropped step becomes 0, which adds nothing to the sum.
    kept = jnp.where(mask[:, :, None], x, 0.0)
    count = mask.sum(axis=1)[:, None]  # (B, 1): kept steps per row
    # A row that keeps nothing has sum 0 and count 0; dividing by 1 returns its zeros.
    return kept.sum(axis=1) / jnp.maximum(count, 1)


def masked_max(x: jax.Array, mask: jax.Array) -> jax.Array:
    # A dropped step becomes -inf, which no kept value loses to. A row that keeps
    # nothing is all -inf, and so is its max.
    return jnp.where(mask[:, :, None], x, -jnp.inf).max(axis=1)
