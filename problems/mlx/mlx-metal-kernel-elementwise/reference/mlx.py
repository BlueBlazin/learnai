import mlx.core as mx

# The body of the kernel function; MLX writes the signature. One thread per element:
# thread `elem` reads a[elem] and b[elem] and writes out[elem]. T is the element type,
# so the arithmetic runs in half precision for float16 inputs.
_SOURCE = """
    uint elem = thread_position_in_grid.x;
    T x = a[elem];
    // x * sigmoid(x). exp(-x) can overflow only for very negative x, where it sits in
    // the denominator and the quotient goes to -0, the right limit.
    out[elem] = x / (T(1) + metal::exp(-x)) * b[elem];
"""

# Built once, at import. ensure_row_contiguous is left at True: MLX copies an input
# that is a strided view, so a[elem] is element `elem` in row-major order.
_KERNEL = mx.fast.metal_kernel(
    name="fused_silu_mul",
    input_names=["a", "b"],
    output_names=["out"],
    source=_SOURCE,
)


def fused_silu_mul(a: mx.array, b: mx.array) -> mx.array:
    shape = a.shape
    if a.ndim == 0:
        # A 0-d input is handed to a kernel by value (`const float& a`), not as a
        # buffer that can be indexed; give it one dimension and take it away again.
        a, b = a.reshape(1), b.reshape(1)
    out = _KERNEL(
        inputs=[a, b],
        template=[("T", a.dtype)],
        grid=(a.size, 1, 1),  # one thread per element, whatever the shape
        threadgroup=(256, 1, 1),
        output_shapes=[a.shape],
        output_dtypes=[a.dtype],
    )[0]
    return out if out.shape == shape else out.reshape(shape)
