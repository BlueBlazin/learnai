import math

import mlx.core as mx


def attention_with_sinks(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    mask: mx.array,
    sinks: mx.array,
) -> mx.array:
    batch, n_heads, n_queries, _ = q.shape
    scores = q @ mx.swapaxes(k, -2, -1) / math.sqrt(q.shape[-1])  # (B, H, T, S)
    scores = mx.where(mask, scores, float("-inf"))  # mask is True where attention is allowed
    # One extra column per row holds the head's sink logit. It is not scaled by
    # 1/sqrt(d) and never masked: it is a logit, not a score against a key.
    sink = mx.broadcast_to(sinks.reshape(1, n_heads, 1, 1), (batch, n_heads, n_queries, 1))
    weights = mx.softmax(mx.concatenate((scores, sink), axis=-1), axis=-1)  # (B, H, T, S + 1)
    # The sink takes its share of the probability and has no value: drop its column.
    return weights[..., :-1] @ v
