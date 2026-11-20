import mlx.core as mx


def partial_rope(x: mx.array, positions: mx.array, base: float, factor: float) -> mx.array:
    """Rotate the first int(D * factor) channels as a half-split RoPE of that width.

    x: float32, shape (B, H, T, D) with D even; D * factor is an even integer r.
    positions: integer tensor, shape (T,). The absolute position of each of the T tokens.
    base: float > 1. Pair i is channels (i, i + r/2) and turns by
        positions[t] * base ** (-2 * i / r) radians.
    factor: float in (0, 1].
    Returns float32, shape (B, H, T, D); channels r and above are returned unchanged.
    """
    raise NotImplementedError


def proportional_rope(x: mx.array, positions: mx.array, base: float, factor: float) -> mx.array:
    """Rotate the first int(factor * D // 2) half-split pairs of the whole head.

    x: float32, shape (B, H, T, D) with D even.
    positions: integer tensor, shape (T,). The absolute position of each of the T tokens.
    base: float > 1. Pair i is channels (i, i + D/2) and turns by
        positions[t] * base ** (-2 * i / D) radians.
    factor: float in (0, 1].
    Returns float32, shape (B, H, T, D); the other pairs are returned unchanged.
    """
    raise NotImplementedError
