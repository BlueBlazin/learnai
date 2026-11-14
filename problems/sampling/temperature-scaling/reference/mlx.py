import mlx.core as mx


def apply_temperature(logits: mx.array, temperature: mx.array) -> mx.array:
    t = temperature[:, None]  # (B, 1): one temperature per row, broadcast over the vocabulary
    greedy = t == 0
    # Greedy rows are divided by 1 instead of 0, so no inf or NaN is ever computed, not
    # even in a branch that is thrown away below.
    scaled = logits / mx.where(greedy, 1.0, t)
    # Greedy rows: the first maximum (argmax returns the lowest index among ties) keeps
    # logit 0 and every other entry is masked. Softmax of that row is exactly one-hot.
    top = mx.argmax(logits, axis=-1, keepdims=True)
    is_top = mx.arange(logits.shape[-1]) == top
    one_hot = mx.where(is_top, 0.0, -mx.inf)
    return mx.where(greedy, one_hot, scaled)
