# BUG: scatter_penalty uses the accumulating update array.at[...].add, so an id that occurs twice in a row is penalized twice.
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
    return logits.at[rows, ids].add(-value)
