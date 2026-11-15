import jax
import jax.numpy as jnp


def kl_from_logits(p_logits: jax.Array, q_logits: jax.Array) -> jax.Array:
    # Log-probabilities straight from the logits: finite wherever the logit is finite,
    # even when the probability itself underflows.
    log_p = jax.nn.log_softmax(p_logits, axis=-1)
    log_q = jax.nn.log_softmax(q_logits, axis=-1)
    # Zero probabilities are decided by the logits, never by an underflowed exp.
    p_zero = jnp.isneginf(p_logits)
    q_zero = jnp.isneginf(q_logits)
    terms = jnp.exp(log_p) * (log_p - log_q)
    terms = jnp.where(q_zero, jnp.inf, terms)  # q = 0 < p: infinite, however small p is
    terms = jnp.where(p_zero, 0.0, terms)  # p = 0: nothing, even where q = 0 too
    # The exact sum is >= 0; rounding can leave a tiny negative number.
    return jnp.maximum(jnp.sum(terms, axis=-1), 0.0)
