import math

import mlx.core as mx


def mirostat_step(
    logits: mx.array, mu: mx.array, tau: float, eta: float, u: mx.array
) -> tuple[mx.array, mx.array]:
    # Surprise of every token in bits, under the model's own distribution. A masked
    # token has log-probability -inf and surprise +inf.
    log_p = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    surprise = -log_p / math.log(2.0)
    # Keep the tokens that are no more surprising than mu. The most likely token is
    # the least surprising one, so "keep it always" is the same as "keep it when
    # nothing else passes".
    top = mx.argmax(logits, axis=-1, keepdims=True)
    index = mx.arange(logits.shape[-1])
    keep = (surprise <= mu[:, None]) | (index == top)
    # Inverse-CDF sampling over the kept tokens, in token order. Scaling u by the kept
    # mass is the renormalization.
    cdf = mx.cumsum(mx.where(keep, mx.exp(log_p), 0.0), axis=-1)
    token = mx.sum(cdf <= u[:, None] * cdf[:, -1:], axis=-1)
    # Feedback: the surprise that was observed, before truncation, against the target.
    observed = mx.take_along_axis(surprise, token[:, None], axis=-1)[:, 0]
    return token, mu - eta * (observed - tau)
