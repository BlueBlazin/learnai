import jax
import jax.numpy as jnp


def top_p_filter(logits: jax.Array, p: float) -> jax.Array:
    """Nucleus filtering: the renormalized distribution over each row's top-p tokens.

    logits: float32 array of shape (B, V); entries may be -inf (masked).
    p: the probability mass to keep, 0 < p <= 1.
    Returns a float32 array of shape (B, V): probabilities in the original token
    order, zero for removed tokens, each row summing to 1.
    """
    raise NotImplementedError
