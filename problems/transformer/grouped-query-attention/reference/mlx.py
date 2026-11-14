import math

import mlx.core as mx


def gqa(q: mx.array, k: mx.array, v: mx.array, mask: mx.array) -> mx.array:
    # G query heads share each KV head: query head h reads KV head h // G.
    groups = q.shape[1] // k.shape[1]
    # Repeat every KV head G times in place, (B, H_kv, S, d) -> (B, H, S, d), so that
    # the head order is 0, 0, .., 1, 1, .. and lines up with the query heads.
    k = mx.repeat(k, groups, axis=1)
    v = mx.repeat(v, groups, axis=1)
    # From here on it is ordinary multi-head attention.
    scores = q @ mx.swapaxes(k, -2, -1) / math.sqrt(q.shape[-1])  # (B, H, T, S)
    scores = mx.where(mask, scores, float("-inf"))  # mask is True where attention is allowed
    return mx.softmax(scores, axis=-1) @ v
