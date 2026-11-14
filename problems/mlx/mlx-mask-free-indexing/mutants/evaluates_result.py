# BUG: calls mx.eval on the result before returning it "to be safe"; the filter forces an evaluation of its own instead of joining the step's graph.
import math

import mlx.core as mx


def top_k_filter(logits: mx.array, k: int) -> mx.array:
    vocab = logits.shape[-1]
    if k >= vocab:
        return logits
    order = mx.argpartition(-logits, kth=k - 1, axis=-1)
    minus_inf = mx.array(-math.inf, dtype=logits.dtype)
    out = mx.put_along_axis(logits, order[..., k:], minus_inf, axis=-1)
    mx.eval(out)
    return out


def scatter_penalty(logits: mx.array, ids: mx.array, value: float) -> mx.array:
    picked = mx.take_along_axis(logits, ids, axis=-1)
    out = mx.put_along_axis(logits, ids, picked - value, axis=-1)
    mx.eval(out)
    return out
