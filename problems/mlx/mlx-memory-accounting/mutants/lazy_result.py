# BUG: the result of fn is never evaluated inside the measurement; a lazy result has allocated nothing, so peak and active read as (almost) zero.
import math

import mlx.core as mx


def predicted_bytes(shapes: list[tuple[int, ...]], dtype: mx.Dtype) -> int:
    return sum(math.prod(shape) for shape in shapes) * dtype.size


def measure(fn, *args) -> dict:
    mx.eval(args)
    mx.synchronize()
    mx.clear_cache()
    mx.reset_peak_memory()
    baseline = mx.get_active_memory()

    result = fn(*args)
    mx.synchronize()

    return {
        "result": result,
        "peak_bytes": mx.get_peak_memory() - baseline,
        "active_bytes": mx.get_active_memory() - baseline,
        "cache_bytes": mx.get_cache_memory(),
    }
