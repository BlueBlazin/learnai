import jax
import jax.numpy as jnp


def top_k_filter(logits: jax.Array, k: int) -> jax.Array:
    if k <= 0 or k >= logits.shape[-1]:
        return logits  # k = 0 means "top-k is off"; k >= V keeps everything
    # argsort is stable, so sorting the negated logits ranks equal logits by token
    # index, lowest first: order[b, r] is the token at rank r of row b.
    order = jnp.argsort(-logits, axis=-1)
    # The argsort of a permutation is its inverse: rank[b, v] is the rank of token v.
    rank = jnp.argsort(order, axis=-1)
    # Ranks are distinct, so exactly k entries per row have rank < k.
    return jnp.where(rank < k, logits, -jnp.inf)
