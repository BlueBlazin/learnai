import mlx.core as mx


def _log_softmax(x: mx.array) -> mx.array:
    # Shift by the row maximum first. `x - logsumexp(x)` would subtract two numbers of
    # the size of the logits and lose their low digits (about 1e-3 at logits near 1e4).
    shifted = x - mx.max(x, axis=-1, keepdims=True)
    return shifted - mx.logsumexp(shifted, axis=-1, keepdims=True)


def kl_from_logits(p_logits: mx.array, q_logits: mx.array) -> mx.array:
    # Log-probabilities straight from the logits: finite wherever the logit is finite,
    # even when the probability itself underflows.
    log_p = _log_softmax(p_logits)
    log_q = _log_softmax(q_logits)
    # Zero probabilities are decided by the logits, never by an underflowed exp.
    p_zero = mx.isneginf(p_logits)
    q_zero = mx.isneginf(q_logits)
    terms = mx.exp(log_p) * (log_p - log_q)
    terms = mx.where(q_zero, mx.inf, terms)  # q = 0 < p: infinite, however small p is
    terms = mx.where(p_zero, 0.0, terms)  # p = 0: nothing, even where q = 0 too
    # The exact sum is >= 0; rounding can leave a tiny negative number.
    return mx.maximum(mx.sum(terms, axis=-1), 0.0)
