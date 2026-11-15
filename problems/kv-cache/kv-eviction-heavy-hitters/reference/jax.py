import jax
import jax.numpy as jnp

Cache = tuple[jax.Array, jax.Array]  # (k, v)


def h2o_step(
    cache: Cache,
    scores: jax.Array | None,
    attn_weights: jax.Array,
    budget: int,
    recent: int,
) -> tuple[Cache, jax.Array, jax.Array]:
    k, v = cache
    b, h, t, s = attn_weights.shape
    # What each cached token received in this step: the column sums over the new queries.
    received = attn_weights.sum(axis=2)
    if scores is None:
        scores = received
    else:
        # The T new tokens start at 0; the older ones keep what they have accumulated.
        scores = jnp.concatenate((scores, jnp.zeros((b, h, t))), axis=2) + received
    if s <= budget:
        kept = jnp.broadcast_to(jnp.arange(s), (b, h, s))
    else:
        old = s - recent  # positions 0 .. old - 1 compete; the last `recent` are safe
        # argsort is stable: sorting the negated scores ranks ties by position, oldest first.
        ranked = jnp.argsort(-scores[:, :, :old], axis=-1)
        heavy = jnp.sort(ranked[:, :, : budget - recent], axis=-1)  # back in time order
        window = jnp.broadcast_to(jnp.arange(old, s), (b, h, recent))
        kept = jnp.concatenate((heavy, window), axis=-1)
    # Every head keeps its own tokens: gather along the sequence axis with per-head indices.
    index = kept[..., None]
    cache = (jnp.take_along_axis(k, index, axis=2), jnp.take_along_axis(v, index, axis=2))
    return cache, jnp.take_along_axis(scores, kept, axis=2), kept
