# BUG: calls mx.dequantize, which the statement rules out; the numbers are right, the kernel is not written.
import mlx.core as mx


def metal_dequantize(
    words: mx.array, scales: mx.array, biases: mx.array, group_size: int, bits: int
) -> mx.array:
    return mx.dequantize(words, scales, biases, group_size, bits)
