import mlx.core as mx


def attend(q: mx.array, k: mx.array, v: mx.array, n_cached: int, causal: bool) -> mx.array:
    steps, slots = q.shape[2], k.shape[2]
    used = n_cached + steps
    if used == slots:
        # Every slot is valid. "causal" aligns the last query with the last key, which
        # is exactly this layout; one query, or a non-causal block, needs no mask.
        mask = "causal" if causal and steps > 1 else None
    else:
        # Unused capacity follows the valid slots: say which slots each query may read.
        # A boolean mask (True = may attend) has no dtype to keep in step with q.
        last = n_cached + mx.arange(steps)[:, None] if causal else used - 1
        mask = mx.arange(slots)[None, :] <= last
    # k and v go in as they are: with N_kv < N_q the kernel pairs each key/value head
    # with a group of consecutive query heads itself.
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=q.shape[-1] ** -0.5, mask=mask)
