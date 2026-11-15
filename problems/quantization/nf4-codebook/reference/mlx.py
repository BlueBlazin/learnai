import mlx.core as mx

OFFSET = 0.9677083  # the outermost quantile; the inverse CDF there is about 1.848


def _ndtri(p: mx.array) -> mx.array:
    # The inverse normal CDF, from the inverse error function.
    return (2.0**0.5) * mx.erfinv(2.0 * p - 1.0)


def nf4_levels() -> mx.array:
    # Eight positive levels and seven negative ones: evenly spaced probabilities from
    # OFFSET down to (but excluding) 0.5, pushed through the inverse normal CDF.
    positive = _ndtri(mx.linspace(OFFSET, 0.5, 9)[:-1])
    negative = -_ndtri(mx.linspace(OFFSET, 0.5, 8)[:-1])
    # Ascending order, with an exact 0 between the two halves.
    levels = mx.concatenate([negative, mx.zeros(1), positive[::-1]])
    # Both ends come from the same quantile, so this maps them to -1 and 1.
    return levels / mx.max(levels)


def nf4_quantize(w: mx.array, block_size: int) -> tuple[mx.array, mx.array]:
    out_dim, in_dim = w.shape
    # Consecutive runs of block_size inputs: (out, n_blocks, block_size).
    blocks = w.reshape(out_dim, in_dim // block_size, block_size)
    absmax = mx.max(mx.abs(blocks), axis=-1)
    # An all-zero block gets scale 1, so that the division below is 0 / 1.
    absmax = mx.where(absmax > 0, absmax, mx.ones_like(absmax))
    normalized = blocks / absmax[..., None]  # in [-1, 1]
    # The nearest level by value: compare with all 16 and take the closest.
    codes = mx.argmin(mx.abs(normalized[..., None] - nf4_levels()), axis=-1)
    return codes.reshape(out_dim, in_dim).astype(mx.uint8), absmax


def nf4_dequantize(codes: mx.array, absmax: mx.array, block_size: int) -> mx.array:
    # A table lookup, then each block is scaled back by its own absmax.
    values = mx.take(nf4_levels(), codes.astype(mx.int32))
    return values * mx.repeat(absmax, block_size, axis=1)
