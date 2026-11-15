import mlx.core as mx


def snapkv_select(attn: mx.array, window: int, budget: int, kernel: int) -> mx.array:
    b, h, length, _ = attn.shape
    if length <= budget:
        return mx.broadcast_to(mx.arange(length), (b, h, length))
    prefix = length - window
    # Vote: what each prefix position receives from the queries of the observation
    # window, per head. Rows before the window and the window's own columns do not count.
    vote = attn[:, :, prefix:, :prefix].sum(axis=2)  # (B, H, prefix)
    # Max-pool along the sequence, stride 1: a position gets the best vote within
    # kernel // 2 of it. Padding with -inf keeps the edges inside the prefix.
    half = kernel // 2
    padded = mx.pad(vote, [(0, 0), (0, 0), (half, half)], constant_values=float("-inf"))
    pooled = mx.stack([padded[..., o : o + prefix] for o in range(kernel)]).max(axis=0)
    # argsort is stable: sorting the negated votes ranks ties by position, lowest first.
    ranked = mx.argsort(-pooled, axis=-1)
    chosen = mx.sort(ranked[..., : budget - window], axis=-1)
    tail = mx.broadcast_to(mx.arange(prefix, length), (b, h, window))
    return mx.concatenate((chosen, tail), axis=-1)
