import math

import mlx.core as mx


def predicted_bytes(shapes: list[tuple[int, ...]], dtype: mx.Dtype) -> int:
    # math.prod(()) is 1: a scalar is one element.
    return sum(math.prod(shape) for shape in shapes) * dtype.size


def measure(fn, *args) -> dict:
    # Lazy arguments would be computed inside the measurement and charged to fn.
    mx.eval(args)
    mx.synchronize()  # nothing from earlier work is still running
    mx.clear_cache()  # freed buffers of earlier work must not be reported as fn's
    mx.reset_peak_memory()  # the high-water mark starts over
    baseline = mx.get_active_memory()  # the arguments and everything else alive

    result = fn(*args)
    mx.eval(result)  # a lazy result has allocated nothing: force the work here
    # Buffers of temporaries are released when the GPU reports completion, which can be
    # a moment after mx.eval returns; wait, or they are still counted as active.
    mx.synchronize()

    return {
        "result": result,
        "peak_bytes": mx.get_peak_memory() - baseline,
        "active_bytes": mx.get_active_memory() - baseline,
        "cache_bytes": mx.get_cache_memory(),
    }
