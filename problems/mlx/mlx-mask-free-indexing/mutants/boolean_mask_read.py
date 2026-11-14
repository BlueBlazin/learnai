# BUG: reads through a boolean mask (logits[keep]) as in NumPy or PyTorch; MLX supports boolean masks for assignment only and raises.
import math

import mlx.core as mx


def top_k_filter(logits: mx.array, k: int) -> mx.array:
    vocab = logits.shape[-1]
    if k >= vocab:
        return logits
    kth = mx.min(mx.topk(logits, k, axis=-1), axis=-1, keepdims=True)
    keep = logits >= kth
    out = mx.full(logits.shape, -math.inf, dtype=logits.dtype)
    out[keep] = logits[keep]
    return out


def scatter_penalty(logits: mx.array, ids: mx.array, value: float) -> mx.array:
    picked = mx.take_along_axis(logits, ids, axis=-1)
    return mx.put_along_axis(logits, ids, picked - value, axis=-1)
