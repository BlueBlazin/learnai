# BUG: lazy arguments are not evaluated before the baseline is taken, so the memory of computing and holding them is charged to fn.
import math

import mlx.core as mx


def predicted_bytes(shapes: list[tuple[int, ...]], dtype: mx.Dtype) -> int:
    return sum(math.prod(shape) for shape in shapes) * dtype.size


def measure(fn, *args) -> dict:
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
