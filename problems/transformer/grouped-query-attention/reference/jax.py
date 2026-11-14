import math

import jax
import jax.numpy as jnp


def gqa(q: jax.Array, k: jax.Array, v: jax.Array, mask: jax.Array) -> jax.Array:
    # G query heads share each KV head: query head h reads KV head h // G.
    groups = q.shape[1] // k.shape[1]
    # Repeat every KV head G times in place, (B, H_kv, S, d) -> (B, H, S, d), so that
    # the head order is 0, 0, .., 1, 1, .. and lines up with the query heads.
    k = jnp.repeat(k, groups, axis=1)
    v = jnp.repeat(v, groups, axis=1)
    # From here on it is ordinary multi-head attention.
    scores = q @ jnp.swapaxes(k, -2, -1) / math.sqrt(q.shape[-1])  # (B, H, T, S)
    scores = jnp.where(mask, scores, -jnp.inf)  # mask is True where attention is allowed
    return jax.nn.softmax(scores, axis=-1) @ v
