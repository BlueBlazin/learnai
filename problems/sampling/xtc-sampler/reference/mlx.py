import mlx.core as mx


def xtc_filter(logits: mx.array, threshold: float, probability: float, u: mx.array) -> mx.array:
    probs = mx.softmax(logits, axis=-1)
    above = probs >= threshold  # the "top choices" of each row
    # XTC acts only when a row has at least two top choices and its coin says so.
    fires = (mx.sum(above, axis=-1, keepdims=True) >= 2) & (u[:, None] < probability)
    # The survivor is the least likely top choice: the smallest logit among them, and
    # of equal logits the one with the highest index.
    floor = mx.min(mx.where(above, logits, mx.inf), axis=-1, keepdims=True)
    index = mx.arange(logits.shape[-1])
    candidate = above & (logits == floor)
    survivor = mx.max(mx.where(candidate, index, -1), axis=-1, keepdims=True)
    # Remove every other top choice. Tokens below the threshold are not touched.
    remove = fires & above & (index != survivor)
    return mx.where(remove, -mx.inf, logits)
