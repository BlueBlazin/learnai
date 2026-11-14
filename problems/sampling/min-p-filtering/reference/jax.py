import jax
import jax.numpy as jnp


def min_p_filter(logits: jax.Array, min_p: float) -> jax.Array:
    probs = jax.nn.softmax(logits, axis=-1)
    # The cutoff is relative to each row's own top probability, so it adapts: a confident
    # row gets a high cutoff and keeps few tokens, a flat row gets a low one and keeps many.
    threshold = min_p * jnp.max(probs, axis=-1, keepdims=True)
    # `>=` keeps the top token itself (and its ties) for every min_p <= 1, and keeps
    # everything when min_p is 0. Removed tokens are masked in logit space.
    return jnp.where(probs >= threshold, logits, -jnp.inf)
