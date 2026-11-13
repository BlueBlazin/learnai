import mlx.core as mx


def softmax(x: mx.array, axis: int = -1) -> mx.array:
    # Shift so the largest logit along the axis is 0: exponents are <= 0 (no overflow)
    # and the largest term is exp(0) = 1 (the denominator is never 0).
    shifted = x - mx.max(x, axis=axis, keepdims=True)
    e = mx.exp(shifted)
    return e / mx.sum(e, axis=axis, keepdims=True)


def log_softmax(x: mx.array, axis: int = -1) -> mx.array:
    # Stay in log space: the sum is in [1, n], so its log is finite and small.
    shifted = x - mx.max(x, axis=axis, keepdims=True)
    return shifted - mx.log(mx.sum(mx.exp(shifted), axis=axis, keepdims=True))
