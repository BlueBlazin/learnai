import mlx.core as mx


def top_p_filter(logits: mx.array, p: float) -> mx.array:
    probs = mx.softmax(logits, axis=-1)
    # MLX's argsort keeps equal elements in index order (it sorts stably), so equal
    # probabilities are ranked by token index, lowest first.
    order = mx.argsort(-probs, axis=-1)
    sorted_probs = mx.take_along_axis(probs, order, axis=-1)
    # Mass of the tokens ranked strictly before each one. A token is kept while that
    # mass is still short of p, so the token that crosses p is kept, and rank 0 (mass
    # before it: exactly 0) always survives.
    mass_before = mx.cumsum(sorted_probs, axis=-1) - sorted_probs
    kept = mx.where(mass_before < p, sorted_probs, 0.0)
    kept = kept / mx.sum(kept, axis=-1, keepdims=True)
    # `order[b, k]` is the token at rank k of row b: scatter sends rank k back there.
    return mx.put_along_axis(mx.zeros_like(probs), order, kept, axis=-1)
