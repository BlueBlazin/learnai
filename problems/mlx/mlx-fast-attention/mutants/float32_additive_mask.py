# BUG: builds an additive mask with mx.where(visible, 0.0, -inf), which is float32 whatever the inputs are; with float16 queries the kernel refuses it.
import mlx.core as mx


def attend(q: mx.array, k: mx.array, v: mx.array, n_cached: int, causal: bool) -> mx.array:
    steps, slots = q.shape[2], k.shape[2]
    used = n_cached + steps
    if used == slots:
        mask = "causal" if causal and steps > 1 else None
    else:
        last = n_cached + mx.arange(steps)[:, None] if causal else used - 1
        mask = mx.where(mx.arange(slots)[None, :] <= last, 0.0, -float("inf"))
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=q.shape[-1] ** -0.5, mask=mask)
