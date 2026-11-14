# BUG: computes attention by hand (matmul, softmax, matmul) instead of calling the fused kernel; the numbers are right, the full (T, S) score matrix is materialized for every head.
import mlx.core as mx


def attend(q: mx.array, k: mx.array, v: mx.array, n_cached: int, causal: bool) -> mx.array:
    steps, slots = q.shape[2], k.shape[2]
    groups = q.shape[1] // k.shape[1]
    k = mx.repeat(k, groups, axis=1)
    v = mx.repeat(v, groups, axis=1)
    scores = (q * q.shape[-1] ** -0.5) @ k.swapaxes(-1, -2)
    last = n_cached + mx.arange(steps)[:, None] if causal else n_cached + steps - 1
    visible = mx.arange(slots)[None, :] <= last
    scores = mx.where(visible, scores, mx.array(-float("inf"), dtype=scores.dtype))
    weights = mx.softmax(scores.astype(mx.float32), axis=-1).astype(q.dtype)
    return weights @ v
