import mlx.core as mx


def top_k_filter(logits: mx.array, k: int) -> mx.array:
    if k <= 0 or k >= logits.shape[-1]:
        return logits  # k = 0 means "top-k is off"; k >= V keeps everything
    # MLX's argsort is stable, so sorting the negated logits ranks equal logits by token
    # index, lowest first: order[b, r] is the token at rank r of row b.
    order = mx.argsort(-logits, axis=-1)
    # The argsort of a permutation is its inverse: rank[b, v] is the rank of token v.
    rank = mx.argsort(order, axis=-1)
    # Ranks are distinct, so exactly k entries per row have rank < k.
    return mx.where(rank < k, logits, -mx.inf)
