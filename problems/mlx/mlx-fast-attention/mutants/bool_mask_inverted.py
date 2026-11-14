# BUG: builds the boolean mask with True for the slots to hide (the masked_fill convention); the kernel reads True as "may attend".
import mlx.core as mx


def attend(q: mx.array, k: mx.array, v: mx.array, n_cached: int, causal: bool) -> mx.array:
    steps, slots = q.shape[2], k.shape[2]
    used = n_cached + steps
    if used == slots:
        mask = "causal" if causal and steps > 1 else None
    else:
        last = n_cached + mx.arange(steps)[:, None] if causal else used - 1
        mask = mx.arange(slots)[None, :] > last
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=q.shape[-1] ** -0.5, mask=mask)
