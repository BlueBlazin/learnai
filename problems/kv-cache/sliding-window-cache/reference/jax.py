import jax
import jax.numpy as jnp

Cache = tuple[jax.Array, jax.Array, int]


def rotating_write(
    cache: Cache, k_new: jax.Array, v_new: jax.Array, max_size: int, keep: int
) -> tuple[Cache, jax.Array, jax.Array]:
    k, v, offset = cache
    window = max_size - keep
    end = offset + k_new.shape[2]
    # Token numbers the cache holds once this write is done, oldest first: the sinks
    # 0 .. keep - 1, then the `window` most recent tokens (none while end <= keep).
    sinks = jnp.arange(min(keep, end))
    recent = jnp.arange(max(keep, end - window), max(keep, end))
    live = jnp.concatenate((sinks, recent))
    # Where each of them lives: a sink in its own slot, the others on a ring of `window`
    # slots behind the sinks. A token's slot never changes.
    slots = jnp.where(live < keep, live, keep + (live - keep) % window)
    # The new tokens are the tail of `live`. New tokens that are already out of the
    # window (a chunk longer than the window) are not in it, and are never written.
    fresh = int((live >= offset).sum())
    # JAX arrays are immutable: `.at[...].set(...)` returns the updated buffer.
    k = k.at[:, :, slots[-fresh:]].set(k_new[:, :, live[-fresh:] - offset])
    v = v.at[:, :, slots[-fresh:]].set(v_new[:, :, live[-fresh:] - offset])
    # Reading the slots in order of token number gives the entries in temporal order.
    return (k, v, end), k[:, :, slots], v[:, :, slots]
