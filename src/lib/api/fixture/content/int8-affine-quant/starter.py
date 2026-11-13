{{import}}


def quantize(x: {{T}}) -> tuple[{{T}}, float, int]:
    """Return (q as int8, scale, zero_point) using the min and max of x as the range."""
    raise NotImplementedError


def dequantize(q: {{T}}, scale: float, zero_point: int) -> {{T}}:
    """Return the float32 reconstruction."""
    raise NotImplementedError
