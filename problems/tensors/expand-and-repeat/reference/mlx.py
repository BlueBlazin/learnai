import mlx.core as mx


def repeat_kv_heads(k: mx.array, n_rep: int) -> mx.array:
    # mx.repeat repeats each entry along the axis in place: groups 0, 1 become heads
    # 0, 0, 1, 1, so head g * n_rep + r is group g. (mx.tile would give 0, 1, 0, 1.)
    return mx.repeat(k, n_rep, axis=1)


def expand_for_beams(x: mx.array, K: int) -> mx.array:
    # The same along the batch axis: rows 0, 1 become rows 0, 0, 1, 1.
    return mx.repeat(x, K, axis=0)


def tile_positions(pos: mx.array, B: int) -> mx.array:
    # Broadcasting aligns shapes from the right, so (T,) broadcasts to (B, T).
    return mx.broadcast_to(pos, (B, pos.shape[0]))
