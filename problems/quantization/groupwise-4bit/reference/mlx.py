import mlx.core as mx


def quantize_groups(w: mx.array, bits: int, group_size: int) -> tuple[mx.array, mx.array, mx.array]:
    out_dim, in_dim = w.shape
    levels = 2**bits - 1  # the largest code: 2**bits codes have 2**bits - 1 steps between them
    # Split every row into consecutive runs of group_size inputs: (out, n_groups, group_size).
    groups = w.reshape(out_dim, in_dim // group_size, group_size)
    lo = mx.min(groups, axis=-1)
    hi = mx.max(groups, axis=-1)
    # A constant group has hi == lo: give it scale 1, so that the division below is 0 / 1.
    scale = mx.where(hi > lo, (hi - lo) / levels, mx.ones_like(hi))
    codes = mx.round((groups - lo[..., None]) / scale[..., None])
    codes = mx.clip(codes, 0, levels).astype(mx.uint8)
    return codes.reshape(out_dim, in_dim), scale, lo
