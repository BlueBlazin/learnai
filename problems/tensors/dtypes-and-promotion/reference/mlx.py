import mlx.core as mx


def dequantize(q: mx.array, scale: float, zero_point: int) -> mx.array:
    # q - zero_point spans [-255, 255], which int8 cannot hold, and an int8 array
    # minus a Python int stays int8. So leave int8 before subtracting. float32 holds
    # every integer of that range exactly.
    return (q.astype(mx.float32) - zero_point) * scale


def saturating_add_u8(a: mx.array, b: mx.array) -> mx.array:
    # The sum reaches 510. Add in int16, where it fits, clamp there, and only then
    # return to uint8: after the clamp every value is in range, so the cast is exact.
    total = a.astype(mx.int16) + b.astype(mx.int16)
    return mx.minimum(total, 255).astype(mx.uint8)


def block_and_offset(pos: mx.array, size: int) -> tuple[mx.array, mx.array]:
    # Floor division rounds toward negative infinity, and remainder is the one that
    # goes with it: block * size + offset == pos, with 0 <= offset < size.
    return mx.floor_divide(pos, size), mx.remainder(pos, size)
