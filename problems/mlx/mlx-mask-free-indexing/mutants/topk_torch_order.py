# BUG: takes the last element of mx.topk as the k-th largest logit, as with torch.topk (sorted descending); in MLX that element is usually the largest, so one token survives.
import math

import mlx.core as mx


def top_k_filter(logits: mx.array, k: int) -> mx.array:
    vocab = logits.shape[-1]
    if k >= vocab:
        return logits
    kth = mx.topk(logits, k, axis=-1)[..., -1:]
    return mx.where(logits >= kth, logits, -math.inf)


def scatter_penalty(logits: mx.array, ids: mx.array, value: float) -> mx.array:
    picked = mx.take_along_axis(logits, ids, axis=-1)
    return mx.put_along_axis(logits, ids, picked - value, axis=-1)
