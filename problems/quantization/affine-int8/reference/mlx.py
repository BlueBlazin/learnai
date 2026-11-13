import mlx.core as mx

QMIN, QMAX = -128, 127


def quant_params(x: mx.array) -> tuple[mx.array, mx.array]:
    # Widen the range to include 0, so that 0.0 lands exactly on a grid point.
    lo = mx.minimum(mx.min(x), 0.0)
    hi = mx.maximum(mx.max(x), 0.0)
    # 256 codes have 255 steps between them. An all-zero tensor gets scale 1.
    scale = mx.where(hi > lo, (hi - lo) / (QMAX - QMIN), mx.ones_like(hi))
    # lo must map to QMIN: round(lo / scale) + zero_point = QMIN.
    zero_point = mx.clip(QMIN - mx.round(lo / scale), QMIN, QMAX)
    return scale, zero_point.astype(mx.int32)


def quantize(x: mx.array, scale: mx.array, zero_point: mx.array) -> mx.array:
    q = mx.round(x / scale) + zero_point
    # Clamp while still in float: do not rely on what an out-of-range cast does.
    return mx.clip(q, QMIN, QMAX).astype(mx.int8)


def dequantize(q: mx.array, scale: mx.array, zero_point: mx.array) -> mx.array:
    # Widen before subtracting: q - zero_point spans [-255, 255], which overflows int8.
    return (q.astype(mx.int32) - zero_point).astype(mx.float32) * scale
