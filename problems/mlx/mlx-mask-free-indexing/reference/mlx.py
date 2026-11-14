import math

import mlx.core as mx


def top_k_filter(logits: mx.array, k: int) -> mx.array:
    vocab = logits.shape[-1]
    if k >= vocab:  # a shape, not a value: nothing is evaluated to decide this
        return logits
    # Partition the negated logits around position k - 1: the first k indices are the k
    # largest logits, the rest are everything to remove. Order inside each part is
    # undefined, and it does not matter here.
    order = mx.argpartition(-logits, kth=k - 1, axis=-1)
    minus_inf = mx.array(-math.inf, dtype=logits.dtype)
    return mx.put_along_axis(logits, order[..., k:], minus_inf, axis=-1)


def scatter_penalty(logits: mx.array, ids: mx.array, value: float) -> mx.array:
    # Gather, modify, scatter. Repeated ids write the same number to the same place, so
    # it does not matter which write wins and the token is penalized once.
    picked = mx.take_along_axis(logits, ids, axis=-1)
    return mx.put_along_axis(logits, ids, picked - value, axis=-1)
