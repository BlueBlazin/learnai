# BUG: calls the kernel once per row from a Python loop and concatenates; the values are right, but a layer with thousands of rows pays thousands of kernel launches.
import mlx.core as mx

_SOURCE = """
    uint col = thread_position_in_grid.x;
    uint per_word = 32 / BITS;
    uint n_in = words_shape[1] * per_word;
    uint word = words[col / per_word];
    uint code = (word >> (BITS * (col % per_word))) & ((1u << BITS) - 1u);
    uint group = col / (n_in / scales_shape[1]);
    out[col] = T(fma(float(code), float(scales[group]), float(biases[group])));
"""

_KERNEL = mx.fast.metal_kernel(
    name="metal_dequantize_row",
    input_names=["words", "scales", "biases"],
    output_names=["out"],
    source=_SOURCE,
)


def metal_dequantize(
    words: mx.array, scales: mx.array, biases: mx.array, group_size: int, bits: int
) -> mx.array:
    n_out, n_groups = scales.shape
    n_in = n_groups * group_size
    rows = [
        _KERNEL(
            inputs=[words[r : r + 1], scales[r : r + 1], biases[r : r + 1]],
            template=[("T", scales.dtype), ("BITS", bits)],
            grid=(n_in, 1, 1),
            threadgroup=(32, 1, 1),
            output_shapes=[(1, n_in)],
            output_dtypes=[scales.dtype],
        )[0]
        for r in range(n_out)
    ]
    return mx.concatenate(rows, axis=0)
