import math

import mlx.core as mx


def _rotate(x: mx.array, positions: mx.array, rope_dim: int, base: float) -> mx.array:
    """Rotate the trailing `rope_dim` channels of `x` (..., N, d), row n by positions[n]."""
    freqs = mx.power(base, -mx.arange(0, rope_dim, 2, dtype=mx.float32) / rope_dim)
    angles = positions.astype(mx.float32)[:, None] * freqs  # (N, rope_dim / 2)
    cos, sin = mx.cos(angles), mx.sin(angles)
    split = x.shape[-1] - rope_dim
    plain, tail = x[..., :split], x[..., split:]
    # Interleaved pairs inside the slice: (0, 1), (2, 3), ...
    a, b = tail[..., 0::2], tail[..., 1::2]
    rotated = mx.stack((a * cos - b * sin, a * sin + b * cos), axis=-1).reshape(tail.shape)
    return mx.concatenate((plain, rotated), axis=-1)


def shared_kv_attention(
    q: mx.array,
    kv: mx.array,
    pos_q: mx.array,
    pos_k: mx.array,
    mask: mx.array,
    rope_dim: int,
    base: float,
) -> mx.array:
    q = _rotate(q, pos_q, rope_dim, base)
    kv = _rotate(kv, pos_k, rope_dim, base)  # one tensor: the key and the value
    scores = q @ kv.T / math.sqrt(q.shape[-1])  # (H, T, S)
    scores = mx.where(mask, scores, -mx.inf)
    out = mx.softmax(scores, axis=-1) @ kv  # (H, T, d): a mix of rotated rows
    # Each value carries the rotation of its own position s. Turning the sum back by the
    # query's position t leaves every term rotated by s - t: a relative angle.
    return _rotate(out, -pos_q, rope_dim, base)
