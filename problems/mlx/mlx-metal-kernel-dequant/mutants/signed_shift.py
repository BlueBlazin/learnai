# BUG: isolates the code with a left shift followed by a right shift on a signed int; the right shift copies the sign bit, so every code with its top bit set comes out negative.
import mlx.core as mx

# One thread per weight, on a two-dimensional grid: x is the column (the input
# feature), y is the row. BITS is a template integer, so the divisions and the mask
# are constants in the compiled kernel; the widths are read from the shapes MLX passes
# when the body mentions them.
_SOURCE = """
    uint col = thread_position_in_grid.x;
    uint row = thread_position_in_grid.y;
    uint per_word = 32 / BITS;              // codes in one uint32 word
    uint n_words = words_shape[1];          // words in one row
    uint n_groups = scales_shape[1];        // groups in one row
    uint n_in = n_words * per_word;         // weights in one row

    // Code `col` of the row: word col / per_word, lowest bits first.
    int word = int(words[row * n_words + col / per_word]);
    int code = (word << (32 - BITS - BITS * (col % per_word))) >> (32 - BITS);

    // The scale and bias of this row's group: n_in / n_groups consecutive columns.
    uint group = row * n_groups + col / (n_in / n_groups);
    float w = fma(float(code), float(scales[group]), float(biases[group]));
    out[row * n_in + col] = T(w);
"""

_KERNEL = mx.fast.metal_kernel(
    name="metal_dequantize",
    input_names=["words", "scales", "biases"],
    output_names=["out"],
    source=_SOURCE,
)


def metal_dequantize(
    words: mx.array, scales: mx.array, biases: mx.array, group_size: int, bits: int
) -> mx.array:
    n_out, n_groups = scales.shape
    n_in = n_groups * group_size
    return _KERNEL(
        inputs=[words, scales, biases],
        # T is the element type of the result; the arithmetic is done in float.
        template=[("T", scales.dtype), ("BITS", bits)],
        grid=(n_in, n_out, 1),
        threadgroup=(32, 8, 1),
        output_shapes=[(n_out, n_in)],
        output_dtypes=[scales.dtype],
    )[0]
