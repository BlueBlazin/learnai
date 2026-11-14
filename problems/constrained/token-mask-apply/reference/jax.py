import jax
import jax.numpy as jnp


def mask_logits(logits: jax.Array, allowed: jax.Array) -> jax.Array:
    # A row with no allowed token has no valid next token: fail loudly instead of
    # handing an all -inf row to softmax, which would turn it into NaN.
    if not bool(jnp.all(jnp.any(allowed, axis=-1))):
        raise ValueError("a row of `allowed` has no allowed token")
    # -inf, not a large negative number: exp(-inf) is exactly 0 for any other logits.
    return jnp.where(allowed, logits, -jnp.inf)


def masked_probs(logits: jax.Array, allowed: jax.Array) -> jax.Array:
    # Mask first, normalize second: the softmax then runs over the allowed tokens only,
    # and its shift by the row maximum uses the largest *allowed* logit.
    return jax.nn.softmax(mask_logits(logits, allowed), axis=-1)
