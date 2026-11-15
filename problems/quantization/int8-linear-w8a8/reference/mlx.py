import mlx.core as mx

QMIN, QMAX = -128, 127


def int8_linear(
    x: mx.array, w_q: mx.array, w_scale: mx.array, bias: mx.array
) -> tuple[mx.array, mx.array]:
    # One affine grid per token (row of x), widened to include 0 so that 0.0 is a code.
    lo = mx.minimum(mx.min(x, axis=1, keepdims=True), 0.0)
    hi = mx.maximum(mx.max(x, axis=1, keepdims=True), 0.0)
    x_scale = mx.where(hi > lo, (hi - lo) / (QMAX - QMIN), mx.ones_like(hi))
    zero_point = mx.clip(QMIN - mx.round(lo / x_scale), QMIN, QMAX).astype(mx.int32)
    x_q = mx.clip(mx.round(x / x_scale) + zero_point, QMIN, QMAX).astype(mx.int8)

    # The integer part of the layer. Widen before multiplying: one product of two int8
    # codes already needs 15 bits, and the sum over `in` of them needs more.
    # mx.matmul only takes floating-point inputs, so the int32 products are formed by
    # broadcasting (n, 1, in) against (1, out, in) and summed over the input axis.
    w32 = w_q.astype(mx.int32)
    acc = (x_q.astype(mx.int32)[:, None, :] * w32[None, :, :]).sum(axis=-1)  # (n, out), int32

    # x_q stands for x_scale * (x_q - zero_point), so the product owes a correction:
    # sum_j (x_q - z) * w_q = acc - z * sum_j w_q. The row sums of w_q do not depend on x.
    corrected = acc - zero_point * w32.sum(axis=1)

    # Back to real numbers once, at the end: one scale per token times one per output.
    y = corrected.astype(mx.float32) * (x_scale * w_scale) + bias
    return y, acc
