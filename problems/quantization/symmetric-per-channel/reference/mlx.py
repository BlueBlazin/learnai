import mlx.core as mx


def quantize_weight(w: mx.array, bits: int) -> tuple[mx.array, mx.array]:
    # The largest code. The grid is symmetric, [-qmax, qmax], so -2**(bits-1) is unused.
    qmax = 2 ** (bits - 1) - 1
    # One range per output row: reduce over the input axis.
    max_abs = mx.max(mx.abs(w), axis=1)
    # An all-zero row gets scale 1, so that nothing divides by zero.
    scale = mx.where(max_abs > 0, max_abs / qmax, mx.ones_like(max_abs))
    # scale has shape (out,): add an axis so it lines up with the rows, not the columns.
    q = mx.clip(mx.round(w / scale[:, None]), -qmax, qmax)
    return q.astype(mx.int8), scale


def dequantize(q: mx.array, scale: mx.array) -> mx.array:
    # No zero-point to subtract: code 0 is the real number 0.
    return q.astype(mx.float32) * scale[:, None]
