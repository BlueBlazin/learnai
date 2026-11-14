# BUG: passes mask="causal" whenever causal is True; the string aligns the last query with the last slot of the buffer, which is wrong as soon as the buffer has unused capacity.
import mlx.core as mx


def attend(q: mx.array, k: mx.array, v: mx.array, n_cached: int, causal: bool) -> mx.array:
    steps = q.shape[2]
    mask = "causal" if causal and steps > 1 else None
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=q.shape[-1] ** -0.5, mask=mask)
