{{import}}


def apply_rope(x: {{T}}, positions: {{T}}, base: float = 10000.0) -> {{T}}:
    """
    x: (B, H, T, d) with d even.    positions: (T,) integer positions.
    Returns x rotated, same shape.
    """
    raise NotImplementedError
