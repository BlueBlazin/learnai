import mlx.core as mx


def min_p_filter(logits: mx.array, min_p: float) -> mx.array:
    probs = mx.softmax(logits, axis=-1)
    # The cutoff is relative to each row's own top probability, so it adapts: a confident
    # row gets a high cutoff and keeps few tokens, a flat row gets a low one and keeps many.
    threshold = min_p * mx.max(probs, axis=-1, keepdims=True)
    # `>=` keeps the top token itself (and its ties) for every min_p <= 1, and keeps
    # everything when min_p is 0. Removed tokens are masked in logit space.
    return mx.where(probs >= threshold, logits, -mx.inf)
