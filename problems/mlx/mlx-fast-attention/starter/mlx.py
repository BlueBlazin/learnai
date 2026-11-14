import mlx.core as mx


def attend(q: mx.array, k: mx.array, v: mx.array, n_cached: int, causal: bool) -> mx.array:
    """Attention of T new tokens over a key/value buffer, with MLX's fused kernel.

    q: (B, N_q, T, D) queries of the new tokens.
    k, v: (B, N_kv, S, D) buffers with S >= n_cached + T. Slots [0, n_cached) hold the
        cached tokens, slots [n_cached, n_cached + T) the new tokens, and any slots
        after that are unused capacity with arbitrary contents. N_kv divides N_q.
    n_cached: number of cached tokens (a Python int, >= 0).
    causal: if True, new token i reads slots 0 .. n_cached + i; if False, every new
        token reads all n_cached + T valid slots.
    Returns (B, N_q, T, D) in the dtype of q, computed by exactly one call of
    mx.fast.scaled_dot_product_attention with scale 1 / sqrt(D).
    """
    raise NotImplementedError
