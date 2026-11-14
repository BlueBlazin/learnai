# BUG: with causal=False no mask is passed at all, so the unused capacity of the buffer is read like real tokens.
import mlx.core as mx


def attend(q: mx.array, k: mx.array, v: mx.array, n_cached: int, causal: bool) -> mx.array:
    steps, slots = q.shape[2], k.shape[2]
    used = n_cached + steps
    if not causal:
        mask = None
    elif used == slots:
        mask = "causal" if steps > 1 else None
    else:
        mask = mx.arange(slots)[None, :] <= n_cached + mx.arange(steps)[:, None]
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=q.shape[-1] ** -0.5, mask=mask)
