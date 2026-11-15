import jax
import jax.numpy as jnp


def kl_from_logits(p_logits: jax.Array, q_logits: jax.Array) -> jax.Array:
    """KL(softmax(p_logits) || softmax(q_logits)) along the last axis, in nats.

    p_logits, q_logits: float32 arrays of the same shape (..., V); entries are finite
        or -inf (a masked token, probability exactly 0). p is the reference distribution.
    A term is 0 where p is 0, and +inf where q is 0 and p is not (p_logit finite).
    Returns a float32 array of shape (...): each value is >= 0 or +inf, never NaN.
    """
    raise NotImplementedError
