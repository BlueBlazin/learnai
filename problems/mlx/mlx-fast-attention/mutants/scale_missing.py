# BUG: passes scale=1.0 instead of 1 / sqrt(D); the kernel has no default and does not scale by itself.
import mlx.core as mx


def attend(q: mx.array, k: mx.array, v: mx.array, n_cached: int, causal: bool) -> mx.array:
    steps, slots = q.shape[2], k.shape[2]
    used = n_cached + steps
    if used == slots:
        mask = "causal" if causal and steps > 1 else None
    else:
        last = n_cached + mx.arange(steps)[:, None] if causal else used - 1
        mask = mx.arange(slots)[None, :] <= last
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0, mask=mask)
