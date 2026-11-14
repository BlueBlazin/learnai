import jax
import jax.numpy as jnp

from whetstone.kit import ops


def attend_step(
    q_new: jax.Array,
    k_new: jax.Array,
    v_new: jax.Array,
    cache: tuple[jax.Array, jax.Array] | None,
) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
    if cache is None:
        k, v = k_new, v_new
    else:
        # Append along the sequence axis (2), old entries first. `concatenate` builds
        # new arrays, so the cache that was passed in is left as it was.
        k = jnp.concatenate((cache[0], k_new), axis=2)
        v = jnp.concatenate((cache[1], v_new), axis=2)
    new, total = q_new.shape[2], k.shape[2]
    past = total - new
    # New token i sits at absolute position past + i and may see keys 0 .. past + i:
    # every cached key, and the new keys up to and including its own.
    visible = jnp.arange(total)[None, :] <= past + jnp.arange(new)[:, None]  # (T, S + T)
    return ops.attention(q_new, k, v, visible), (k, v)
