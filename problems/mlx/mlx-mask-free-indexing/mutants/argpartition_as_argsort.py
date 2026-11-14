# BUG: treats mx.argpartition as a full sort and partitions around position 0; only the single largest logit is then guaranteed to come first (the Metal backend happens to sort the whole row, the CPU backend does not).
import math

import mlx.core as mx


def top_k_filter(logits: mx.array, k: int) -> mx.array:
    vocab = logits.shape[-1]
    if k >= vocab:
        return logits
    order = mx.argpartition(-logits, kth=0, axis=-1)
    minus_inf = mx.array(-math.inf, dtype=logits.dtype)
    return mx.put_along_axis(logits, order[..., k:], minus_inf, axis=-1)


def scatter_penalty(logits: mx.array, ids: mx.array, value: float) -> mx.array:
    picked = mx.take_along_axis(logits, ids, axis=-1)
    return mx.put_along_axis(logits, ids, picked - value, axis=-1)
