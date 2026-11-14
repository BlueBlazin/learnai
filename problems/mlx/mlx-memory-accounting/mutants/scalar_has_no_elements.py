# BUG: predicted_bytes multiplies the dimensions starting from 0 elements for an empty shape, so a scalar () counts as 0 bytes instead of one element.
import math

import mlx.core as mx


def predicted_bytes(shapes: list[tuple[int, ...]], dtype: mx.Dtype) -> int:
    return sum(math.prod(shape) if shape else 0 for shape in shapes) * dtype.size


def measure(fn, *args) -> dict:
    mx.eval(args)
    mx.synchronize()
    mx.clear_cache()
    mx.reset_peak_memory()
    baseline = mx.get_active_memory()

    result = fn(*args)
    mx.eval(result)
    mx.synchronize()

    return {
        "result": result,
        "peak_bytes": mx.get_peak_memory() - baseline,
        "active_bytes": mx.get_active_memory() - baseline,
        "cache_bytes": mx.get_cache_memory(),
    }
