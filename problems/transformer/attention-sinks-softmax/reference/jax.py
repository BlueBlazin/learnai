import math

import jax
import jax.numpy as jnp


def attention_with_sinks(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    mask: jax.Array,
    sinks: jax.Array,
) -> jax.Array:
    batch, n_heads, n_queries, _ = q.shape
    scores = q @ jnp.swapaxes(k, -2, -1) / math.sqrt(q.shape[-1])  # (B, H, T, S)
    scores = jnp.where(mask, scores, -jnp.inf)  # mask is True where attention is allowed
    # One extra column per row holds the head's sink logit. It is not scaled by
    # 1/sqrt(d) and never masked: it is a logit, not a score against a key.
    sink = jnp.broadcast_to(sinks.reshape(1, n_heads, 1, 1), (batch, n_heads, n_queries, 1))
    weights = jax.nn.softmax(jnp.concatenate((scores, sink), axis=-1), axis=-1)  # (B, H, T, S + 1)
    # The sink takes its share of the probability and has no value: drop its column.
    return weights[..., :-1] @ v
