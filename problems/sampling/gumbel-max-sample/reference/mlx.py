import mlx.core as mx

U_MIN = 1.1754944e-38  # the smallest positive normal float32
U_MAX = 0.99999994  # the largest float32 below 1


def gumbel_sample(logits: mx.array, uniforms: mx.array) -> mx.array:
    # Keep the uniforms strictly inside (0, 1): at u = 0 the noise would be -inf and at
    # u = 1 it would be +inf, and -inf (a masked logit) plus +inf is NaN.
    u = mx.clip(uniforms, U_MIN, U_MAX)
    # Standard Gumbel noise by inverting its CDF exp(-exp(-g)): g = -log(-log(u)).
    gumbel = -mx.log(-mx.log(u))
    # The noise is finite, so a masked logit stays -inf and can never be the maximum.
    return mx.argmax(logits + gumbel, axis=-1)
