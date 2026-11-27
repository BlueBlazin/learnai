import mlx.core as mx


def rank_descending(x: mx.array) -> mx.array:
    # mx.argsort is stable (documented in MLX 0.32.4) and has no descending argument.
    # Sorting the negated values ascending is a descending sort that still ranks ties
    # lowest index first.
    return mx.argsort(-x, axis=-1)


def inverse_permutation(perm: mx.array) -> mx.array:
    # perm[b, k] is where rank k points. The inverse answers the opposite question,
    # inv[b, perm[b, k]] = k, which is a write of the ranks 0 .. V - 1 through perm.
    ranks = mx.arange(perm.shape[1], dtype=perm.dtype)[None, :]  # broadcast over the rows
    return mx.put_along_axis(mx.zeros_like(perm), perm, ranks, axis=1)


def keep_to_original_order(keep_sorted: mx.array, order: mx.array) -> mx.array:
    # order[b, k] is the token at rank k, so the decision for rank k belongs at column
    # order[b, k]: a write through the permutation. A read through it would sort again.
    return mx.put_along_axis(mx.zeros_like(keep_sorted), order, keep_sorted, axis=1)
