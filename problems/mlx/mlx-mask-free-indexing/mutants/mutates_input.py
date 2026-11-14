# BUG: scatter_penalty assigns into its argument (logits[rows, ids] = ...), which changes the caller's array.
import math

import mlx.core as mx


def top_k_filter(logits: mx.array, k: int) -> mx.array:
    vocab = logits.shape[-1]
    if k >= vocab:
        return logits
    order = mx.argpartition(-logits, kth=k - 1, axis=-1)
    minus_inf = mx.array(-math.inf, dtype=logits.dtype)
    return mx.put_along_axis(logits, order[..., k:], minus_inf, axis=-1)


def scatter_penalty(logits: mx.array, ids: mx.array, value: float) -> mx.array:
    rows = mx.arange(logits.shape[0])[:, None]
    logits[rows, ids] = logits[rows, ids] - value
    return logits
