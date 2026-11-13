import jax
import jax.numpy as jnp


def softmax(x: jax.Array, axis: int = -1) -> jax.Array:
    # Shift so the largest logit along the axis is 0: exponents are <= 0 (no overflow)
    # and the largest term is exp(0) = 1 (the denominator is never 0).
    shifted = x - jnp.max(x, axis=axis, keepdims=True)
    e = jnp.exp(shifted)
    return e / jnp.sum(e, axis=axis, keepdims=True)


def log_softmax(x: jax.Array, axis: int = -1) -> jax.Array:
    # Stay in log space: the sum is in [1, n], so its log is finite and small.
    shifted = x - jnp.max(x, axis=axis, keepdims=True)
    return shifted - jnp.log(jnp.sum(jnp.exp(shifted), axis=axis, keepdims=True))
