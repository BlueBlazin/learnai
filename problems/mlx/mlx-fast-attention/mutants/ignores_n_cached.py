# BUG: the explicit mask is the square causal mask of the new tokens, aligned with slot 0: the cached slots are not accounted for.
import mlx.core as mx


def attend(q: mx.array, k: mx.array, v: mx.array, n_cached: int, causal: bool) -> mx.array:
    steps, slots = q.shape[2], k.shape[2]
    used = n_cached + steps
    if used == slots:
        mask = "causal" if causal and steps > 1 else None
    else:
        last = mx.arange(steps)[:, None] if causal else steps - 1
        mask = mx.arange(slots)[None, :] <= last
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=q.shape[-1] ** -0.5, mask=mask)
