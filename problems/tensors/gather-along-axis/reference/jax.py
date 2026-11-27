import jax
import jax.numpy as jnp


def token_scores(logits: jax.Array, ids: jax.Array) -> jax.Array:
    # take_along_axis wants an index with as many axes as logits: (B, T) -> (B, T, 1).
    # The result has the index's shape, and the size-1 axis that was added is then
    # removed by name.
    return jnp.take_along_axis(logits, ids[:, :, None], axis=2).squeeze(2)


def take_columns(values: jax.Array, idx: jax.Array) -> jax.Array:
    # The index already has the rank of values: out[b, k] = values[b, idx[b, k]].
    return jnp.take_along_axis(values, idx, axis=1)


def reorder_steps(x: jax.Array, perm: jax.Array) -> jax.Array:
    # Along axis 1 this reads out[b, t, d] = x[b, index[b, t, d], d]. The (B, T, 1)
    # index is broadcast along the feature axis, so every feature of a step moves
    # together.
    return jnp.take_along_axis(x, perm[:, :, None], axis=1)
