import math

import mlx.core as mx


def window_mask(T: int, window: int) -> mx.array:
    i = mx.arange(T)[:, None]  # query position, one per row
    j = mx.arange(T)[None, :]  # key position, one per column
    # Causal (j <= i) and at most `window` keys back, counting the query itself:
    # the oldest visible key is i - window + 1.
    return (j <= i) & (j > i - window)


def sliding_window_attention(q: mx.array, k: mx.array, v: mx.array, window: int) -> mx.array:
    scores = q @ mx.swapaxes(k, -2, -1) / math.sqrt(q.shape[-1])  # (B, H, T, T)
    # Hide everything outside the window before the softmax, so that the weights are
    # normalized over the visible keys only.
    visible = window_mask(q.shape[-2], window)
    scores = mx.where(visible, scores, float("-inf"))
    return mx.softmax(scores, axis=-1) @ v
