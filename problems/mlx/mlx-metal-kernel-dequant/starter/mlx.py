import mlx.core as mx


def metal_dequantize(
    words: mx.array, scales: mx.array, biases: mx.array, group_size: int, bits: int
) -> mx.array:
    """Dequantize an MLX affine-quantized matrix with one custom Metal kernel.

    words: uint32 array of shape (out, in * bits / 32); code j of a word is bits
        j * bits to (j + 1) * bits - 1, so a row holds its `in` codes in order.
    scales, biases: arrays of shape (out, in / group_size) with the same dtype,
        float16, bfloat16, or float32.
    group_size: 32, 64, or 128. bits: 2, 4, or 8.
    Returns a lazy array of shape (out, in) with the dtype of `scales`:
    w[r, c] = code[r, c] * scales[r, c // group_size] + biases[r, c // group_size].
    The only computation in its graph is one kernel built with mx.fast.metal_kernel.
    """
    raise NotImplementedError
