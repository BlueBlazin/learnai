# BUG: no special case for k >= V; the partition position k - 1 is then outside the row.
import math

import mlx.core as mx


def top_k_filter(logits: mx.array, k: int) -> mx.array:
    order = mx.argpartition(-logits, kth=k - 1, axis=-1)
    minus_inf = mx.array(-math.inf, dtype=logits.dtype)
    return mx.put_along_axis(logits, order[..., k:], minus_inf, axis=-1)


def scatter_penalty(logits: mx.array, ids: mx.array, value: float) -> mx.array:
    picked = mx.take_along_axis(logits, ids, axis=-1)
    return mx.put_along_axis(logits, ids, picked - value, axis=-1)
