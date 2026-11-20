import mlx.core as mx


def partial_rope(x: mx.array, positions: mx.array, base: float, factor: float) -> mx.array:
    dim = x.shape[-1]
    r = int(dim * factor)  # rotated channels: the first r
    half = r // 2
    # The rotated block is a complete RoPE of width r: its own frequency ladder.
    freqs = mx.power(base, -mx.arange(0, r, 2, dtype=mx.float32) / r)
    angles = positions.astype(mx.float32)[:, None] * freqs  # (T, r/2)
    cos, sin = mx.cos(angles), mx.sin(angles)
    # Half-split inside the block: pair i is channels (i, i + r/2).
    a, b = x[..., :half], x[..., half:r]
    return mx.concatenate((a * cos - b * sin, a * sin + b * cos, x[..., r:]), axis=-1)


def proportional_rope(x: mx.array, positions: mx.array, base: float, factor: float) -> mx.array:
    dim = x.shape[-1]
    half = dim // 2
    n_rot = int(factor * dim // 2)  # rotated pairs: the first n_rot of dim/2
    # The first n_rot rungs of the full head's ladder; every other pair has frequency 0.
    freqs = mx.power(base, -mx.arange(0, 2 * n_rot, 2, dtype=mx.float32) / dim)
    freqs = mx.concatenate((freqs, mx.zeros(half - n_rot, dtype=mx.float32)))
    angles = positions.astype(mx.float32)[:, None] * freqs  # (T, D/2)
    cos, sin = mx.cos(angles), mx.sin(angles)
    # Half-split over the whole head: pair i is channels (i, i + D/2).
    a, b = x[..., :half], x[..., half:]
    return mx.concatenate((a * cos - b * sin, a * sin + b * cos), axis=-1)
