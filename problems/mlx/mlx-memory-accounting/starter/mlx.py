import math

import mlx.core as mx


def predicted_bytes(shapes: list[tuple[int, ...]], dtype: mx.Dtype) -> int:
    """Bytes needed to store arrays of the given shapes, all of dtype `dtype`.

    shapes: a list of shapes (tuples of ints); `()` is a scalar, a 0 in a shape means
        an empty array, and the list may be empty.
    Returns the total as a Python int.
    """
    raise NotImplementedError


def measure(fn, *args) -> dict:
    """Run `fn(*args)` and report the MLX memory attributable to it.

    Returns {"result": fn's result, evaluated,
             "peak_bytes": the most memory in use at one time during the call, above
                 what was in use before it,
             "active_bytes": what the call left in use (its result) above the same,
             "cache_bytes": freed buffers MLX is holding for reuse afterwards}.
    Memory that the arguments occupy, or that their own evaluation needs, is not
    attributed to `fn`; neither is anything left over from earlier work.
    """
    raise NotImplementedError
