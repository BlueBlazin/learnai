import mlx.core as mx


def mask_logits(logits: mx.array, allowed: mx.array) -> mx.array:
    # A row with no allowed token has no valid next token: fail loudly instead of
    # handing an all -inf row to softmax, which would turn it into NaN.
    if not bool(mx.all(mx.any(allowed, axis=-1))):
        raise ValueError("a row of `allowed` has no allowed token")
    # -inf, not a large negative number: exp(-inf) is exactly 0 for any other logits.
    return mx.where(allowed, logits, -mx.inf)


def masked_probs(logits: mx.array, allowed: mx.array) -> mx.array:
    # Mask first, normalize second: the softmax then runs over the allowed tokens only,
    # and its shift by the row maximum uses the largest *allowed* logit.
    return mx.softmax(mask_logits(logits, allowed), axis=-1)
