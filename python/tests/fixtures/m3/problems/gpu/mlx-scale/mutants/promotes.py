# BUG: multiplies by a float32 array, so a float16 input comes back as float32 (twice the memory).
import mlx.core as mx


def scale(x, factor):
    return x * mx.array(factor, dtype=mx.float32)
