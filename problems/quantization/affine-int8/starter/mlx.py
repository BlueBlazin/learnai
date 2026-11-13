import mlx.core as mx

QMIN, QMAX = -128, 127


def quant_params(x: mx.array) -> tuple[mx.array, mx.array]:
    """Scale and zero-point of the affine int8 grid that covers [min(x), max(x)] and 0.

    x: float32 array of any shape with 1 to 4 dimensions (one grid for the whole array).
    Returns (scale, zero_point): a 0-d float32 array with scale > 0, and a 0-d integer
    array with QMIN <= zero_point <= QMAX.
    """
    raise NotImplementedError


def quantize(x: mx.array, scale: mx.array, zero_point: mx.array) -> mx.array:
    """Map x to int8 codes: clamp(round(x / scale) + zero_point, QMIN, QMAX).

    x: float32 array of any shape. scale: 0-d float32. zero_point: 0-d integer.
    Returns an int8 array of the same shape as x.
    """
    raise NotImplementedError


def dequantize(q: mx.array, scale: mx.array, zero_point: mx.array) -> mx.array:
    """Map int8 codes back to real values: (q - zero_point) * scale.

    q: integer array (int8 in practice) with values in [QMIN, QMAX], any shape.
    Returns a float32 array of the same shape as q.
    """
    raise NotImplementedError
