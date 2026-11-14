import mlx.core as mx


def fused_silu_mul(a: mx.array, b: mx.array) -> mx.array:
    """silu(a) * b for every element, computed by one custom Metal kernel.

    a, b: arrays of the same shape (0 to 4 dimensions, possibly empty, possibly strided
        views) and the same dtype, float16 or float32.
    Returns a lazy array of the same shape and dtype with a[i] / (1 + exp(-a[i])) * b[i]
    at every index i. The only computation in its graph is one kernel built with
    mx.fast.metal_kernel.
    """
    raise NotImplementedError
