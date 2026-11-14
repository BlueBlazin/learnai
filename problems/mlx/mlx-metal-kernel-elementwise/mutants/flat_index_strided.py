# BUG: turns off ensure_row_contiguous but still indexes the inputs with the flat element number; a transposed, stepped, reversed, or broadcast view is read in buffer order, not in element order.
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
    ensure_row_contiguous=False,
)


def fused_silu_mul(a: mx.array, b: mx.array) -> mx.array:
    shape = a.shape
    if a.ndim == 0:
        a, b = a.reshape(1), b.reshape(1)
    out = _KERNEL(
        inputs=[a, b],
        template=[("T", a.dtype)],
        grid=(a.size, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[a.shape],
        output_dtypes=[a.dtype],
    )[0]
    return out if out.shape == shape else out.reshape(shape)
