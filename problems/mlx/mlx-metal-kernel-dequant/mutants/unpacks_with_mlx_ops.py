# BUG: unpacks with MLX shifts and masks and applies repeated scales and biases; the numbers are right, but there is no kernel and the temporaries are several times the size of the result.
import mlx.core as mx


def metal_dequantize(
    words: mx.array, scales: mx.array, biases: mx.array, group_size: int, bits: int
) -> mx.array:
    shifts = mx.arange(32 // bits, dtype=mx.uint32) * bits
    codes = (words[:, :, None] >> shifts) & ((1 << bits) - 1)
    codes = codes.reshape(words.shape[0], -1).astype(mx.float32)
    w = codes * mx.repeat(scales, group_size, axis=1).astype(mx.float32)
    return (w + mx.repeat(biases, group_size, axis=1).astype(mx.float32)).astype(scales.dtype)
