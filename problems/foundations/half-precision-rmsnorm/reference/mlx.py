import math

import mlx.core as mx


def rms_norm_f16(x: mx.array, weight: mx.array, eps: float = 1e-5) -> mx.array:
    # Divide each vector by c = max(max|x|, sqrt(eps)) before squaring anything:
    #   x / sqrt(mean(x^2) + eps) == u / sqrt(mean(u^2) + (sqrt(eps) / c)^2),  u = x / c.
    # Every |u| is at most 1 and sqrt(eps) / c is at most 1, and one of the two terms
    # under the root is at least 1 / d, so nothing overflows, nothing that matters
    # underflows, and an all-zero vector needs no special case (c = sqrt(eps) > 0).
    root_eps = math.sqrt(eps)
    c = mx.maximum(mx.max(mx.abs(x), axis=-1, keepdims=True), root_eps)
    u = x / c
    r = root_eps / c
    return u / mx.sqrt(mx.mean(mx.square(u), axis=-1, keepdims=True) + r * r) * weight
