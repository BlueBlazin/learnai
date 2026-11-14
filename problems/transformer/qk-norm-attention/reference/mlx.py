import math

import mlx.core as mx


def _rms_norm(x: mx.array, weight: mx.array, eps: float) -> mx.array:
    # One root mean square per vector, over the head dimension (the last axis) only.
    mean_square = mx.mean(mx.square(x), axis=-1, keepdims=True)
    return x * mx.rsqrt(mean_square + eps) * weight


def _rotate(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    # RoPE on adjacent pairs; cos and sin are (T, d/2) and broadcast over (B, H).
    x1, x2 = x[..., 0::2], x[..., 1::2]
    rotated = mx.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), axis=-1)
    return rotated.reshape(x.shape)


def qk_norm_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    q_weight: mx.array,
    k_weight: mx.array,
    cos: mx.array,
    sin: mx.array,
    eps: float = 1e-6,
) -> mx.array:
    seq, dim = q.shape[-2], q.shape[-1]
    # Normalize first, rotate second: the gains belong to the unrotated dimensions.
    # The values are neither normalized nor rotated.
    q = _rotate(_rms_norm(q, q_weight, eps), cos, sin)
    k = _rotate(_rms_norm(k, k_weight, eps), cos, sin)
    # The usual scale stays, even though the vectors are normalized.
    scores = q @ mx.swapaxes(k, -2, -1) / math.sqrt(dim)
    future = mx.triu(mx.ones((seq, seq)), k=1) > 0
    scores = mx.where(future, float("-inf"), scores)
    return mx.softmax(scores, axis=-1) @ v
