import mlx.core as mx


def _pair_logprobs(logits: mx.array, tokens: mx.array, mask: mx.array) -> tuple[mx.array, mx.array]:
    """Log-probability of each counted (context t, target t + 1) pair, 0 elsewhere."""
    # logits[:, t] predicts tokens[:, t + 1]: drop the last position's logits (they
    # predict a token that is not in the batch) and the first token (nothing predicts it).
    scores = logits[:, :-1]
    logp = scores - mx.logsumexp(scores, axis=-1, keepdims=True)  # (B, T - 1, V)
    targets = tokens[:, 1:]  # (B, T - 1)
    picked = mx.take_along_axis(logp, targets[:, :, None], axis=2)[:, :, 0]  # (B, T - 1)
    # A pair counts when both of its positions are real tokens.
    counted = mx.logical_and(mask[:, :-1], mask[:, 1:])
    # where, not multiplication: an uncounted entry may be -inf, and 0 * -inf is NaN.
    return mx.where(counted, picked, 0.0), counted


def sequence_logprob(logits: mx.array, tokens: mx.array, mask: mx.array) -> mx.array:
    picked, _ = _pair_logprobs(logits, tokens, mask)
    return picked.sum(axis=1)


def perplexity(logits: mx.array, tokens: mx.array, mask: mx.array) -> mx.array:
    picked, counted = _pair_logprobs(logits, tokens, mask)
    # exp of the mean negative log-probability per counted pair, not of the sum.
    return mx.exp(-picked.sum(axis=1) / counted.sum(axis=1))
