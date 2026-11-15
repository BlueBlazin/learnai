from collections.abc import Iterator

import jax
import jax.numpy as jnp

Block = tuple[jax.Array, jax.Array, jax.Array]


def blockwise_attention(q: jax.Array, kv_blocks: Iterator[Block]) -> jax.Array:
    n_queries, d = q.shape
    # Running state, one row per query. `norm` and `acc` are the softmax denominator
    # and the weighted sum of values, both in units of exp(top): the true sums are
    # norm * exp(top) and acc * exp(top), which are never formed.
    top = jnp.full((n_queries, 1), -jnp.inf)  # largest score seen so far
    norm = jnp.zeros((n_queries, 1))
    acc = 0.0  # becomes (n_queries, D_v) with the first block

    for k, v, mask in kv_blocks:
        scores = jnp.where(mask, (q @ k.T) * d**-0.5, -jnp.inf)
        new_top = jnp.maximum(top, jnp.max(scores, axis=-1, keepdims=True))
        # A query that has not seen a visible key yet still has new_top = -inf, and
        # -inf - (-inf) is NaN. Shift by 0 instead: all its terms are exp(-inf) = 0.
        shift = jnp.where(jnp.isinf(new_top), 0.0, new_top)
        # When the maximum grows, the old sums are in the wrong unit: rescale them by
        # exp(top - new_top) <= 1. For the first visible block this is exp(-inf) = 0.
        rescale = jnp.exp(top - shift)
        weights = jnp.exp(scores - shift)  # every exponent is <= 0
        norm = norm * rescale + jnp.sum(weights, axis=-1, keepdims=True)
        acc = acc * rescale + weights @ v
        top = new_top
        # k, v, mask, and everything computed from them are rebound on the next
        # iteration: nothing of this block is kept.

    return acc / norm
