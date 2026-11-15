import math

import jax
import jax.numpy as jnp


def mirostat_step(
    logits: jax.Array, mu: jax.Array, tau: float, eta: float, u: jax.Array
) -> tuple[jax.Array, jax.Array]:
    # Surprise of every token in bits, under the model's own distribution. A masked
    # token has log-probability -inf and surprise +inf.
    log_p = logits - jax.nn.logsumexp(logits, axis=-1, keepdims=True)
    surprise = -log_p / math.log(2.0)
    # Keep the tokens that are no more surprising than mu. The most likely token is
    # the least surprising one, so "keep it always" is the same as "keep it when
    # nothing else passes".
    top = jnp.argmax(logits, axis=-1, keepdims=True)
    index = jnp.arange(logits.shape[-1])
    keep = (surprise <= mu[:, None]) | (index == top)
    # Inverse-CDF sampling over the kept tokens, in token order. Scaling u by the kept
    # mass is the renormalization.
    cdf = jnp.cumsum(jnp.where(keep, jnp.exp(log_p), 0.0), axis=-1)
    token = jnp.sum(cdf <= u[:, None] * cdf[:, -1:], axis=-1)
    # Feedback: the surprise that was observed, before truncation, against the target.
    observed = jnp.take_along_axis(surprise, token[:, None], axis=-1)[:, 0]
    return token, mu - eta * (observed - tau)
