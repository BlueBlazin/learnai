# BUG: the grid is the size of the first dimension, not the number of elements; for anything but a 1-D array only the first a.shape[0] elements are computed.
import mlx.core as mx

_SOURCE = """
    uint elem = thread_position_in_grid.x;
    T x = a[elem];
    out[elem] = x / (T(1) + metal::exp(-x)) * b[elem];
"""

_KERNEL = mx.fast.metal_kernel(
    name="fused_silu_mul",
    input_names=["a", "b"],
    output_names=["out"],
    source=_SOURCE,
)


def fused_silu_mul(a: mx.array, b: mx.array) -> mx.array:
    shape = a.shape
    if a.ndim == 0:
        a, b = a.reshape(1), b.reshape(1)
    out = _KERNEL(
        inputs=[a, b],
        template=[("T", a.dtype)],
        grid=(a.shape[0], 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[a.shape],
        output_dtypes=[a.dtype],
    )[0]
    return out if out.shape == shape else out.reshape(shape)
