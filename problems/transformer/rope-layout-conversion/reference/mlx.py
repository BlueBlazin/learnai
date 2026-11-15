import mlx.core as mx


def _cos_sin(dim: int, pos: mx.array, theta: float) -> tuple[mx.array, mx.array]:
    """cos and sin of pos[t] * theta^(-2i / dim), shape (T, dim / 2). Both layouts use it."""
    freqs = mx.power(theta, -mx.arange(0, dim, 2, dtype=mx.float32) / dim)
    angles = pos.astype(mx.float32)[:, None] * freqs
    return mx.cos(angles), mx.sin(angles)


def rope_interleaved(x: mx.array, pos: mx.array, theta: float = 10000.0) -> mx.array:
    cos, sin = _cos_sin(x.shape[-1], pos, theta)
    # Pair i is features (2i, 2i + 1).
    a, b = x[..., 0::2], x[..., 1::2]
    rotated = mx.stack((a * cos - b * sin, a * sin + b * cos), axis=-1)
    return rotated.reshape(x.shape)  # (..., D/2, 2) -> (..., D): pairs back side by side


def rope_half(x: mx.array, pos: mx.array, theta: float = 10000.0) -> mx.array:
    cos, sin = _cos_sin(x.shape[-1], pos, theta)
    # Pair i is features (i, i + D/2): first members in the first half, second in the second.
    half = x.shape[-1] // 2
    a, b = x[..., :half], x[..., half:]
    return mx.concatenate((a * cos - b * sin, a * sin + b * cos), axis=-1)


def permute_qk(w: mx.array, d_head: int, n_heads: int) -> mx.array:
    # Rows are output features. Within each head, row 2i + c (pair i, member c) of the
    # interleaved layout moves to row c * d_head/2 + i of the half-split layout.
    pairs = w.reshape(n_heads, d_head // 2, 2, w.shape[-1])  # [head, pair, member, in]
    return mx.swapaxes(pairs, 1, 2).reshape(w.shape)  # [head, member, pair, in], flattened
