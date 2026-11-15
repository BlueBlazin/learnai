from collections.abc import Iterator

import mlx.core as mx

Block = tuple[mx.array, mx.array, mx.array]


def blockwise_attention(q: mx.array, kv_blocks: Iterator[Block]) -> mx.array:
    n_queries, d = q.shape
    # Running state, one row per query. `norm` and `acc` are the softmax denominator
    # and the weighted sum of values, both in units of exp(top): the true sums are
    # norm * exp(top) and acc * exp(top), which are never formed.
    top = mx.full((n_queries, 1), -mx.inf)  # largest score seen so far
    norm = mx.zeros((n_queries, 1))
    acc = 0.0  # becomes (n_queries, D_v) with the first block

    for k, v, mask in kv_blocks:
        scores = mx.where(mask, (q @ k.T) * d**-0.5, -mx.inf)
        new_top = mx.maximum(top, mx.max(scores, axis=-1, keepdims=True))
        # A query that has not seen a visible key yet still has new_top = -inf, and
        # -inf - (-inf) is NaN. Shift by 0 instead: all its terms are exp(-inf) = 0.
        shift = mx.where(mx.isinf(new_top), 0.0, new_top)
        # When the maximum grows, the old sums are in the wrong unit: rescale them by
        # exp(top - new_top) <= 1. For the first visible block this is exp(-inf) = 0.
        rescale = mx.exp(top - shift)
        weights = mx.exp(scores - shift)  # every exponent is <= 0
        norm = norm * rescale + mx.sum(weights, axis=-1, keepdims=True)
        acc = acc * rescale + weights @ v
        top = new_top
        # MLX is lazy: without this, the graph of the result would keep every block's
        # arrays alive until the end. Evaluating the state lets each block go.
        mx.eval(top, norm, acc)

    return acc / norm
