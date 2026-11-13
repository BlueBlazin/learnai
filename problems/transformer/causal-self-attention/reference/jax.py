import math

import jax
import jax.numpy as jnp


def causal_self_attention(
    x: jax.Array,
    w_q: jax.Array,
    w_k: jax.Array,
    w_v: jax.Array,
    w_o: jax.Array,
) -> jax.Array:
    seq = x.shape[1]
    # Weights are (out, in), so a projection is x @ W^T: (B, T, d) -> (B, T, out).
    q, k, v = x @ w_q.T, x @ w_k.T, x @ w_v.T
    # scores[b, i, j] = q[b, i] . k[b, j] / sqrt(d_k): row i is the query, column j the key.
    scores = q @ jnp.swapaxes(k, -2, -1) / math.sqrt(q.shape[-1])
    # Position i may look at j <= i. Mask the logits (not the probabilities) so that
    # the softmax renormalizes over the visible positions only.
    future = jnp.triu(jnp.ones((seq, seq), dtype=bool), k=1)
    scores = jnp.where(future, -jnp.inf, scores)
    attn = jax.nn.softmax(scores, axis=-1)
    return (attn @ v) @ w_o.T
