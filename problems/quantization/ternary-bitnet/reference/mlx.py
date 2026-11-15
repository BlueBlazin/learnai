import mlx.core as mx


def ternary_quantize(w: mx.array) -> tuple[mx.array, mx.array]:
    # The absmean: one scale for the matrix, the mean magnitude of all elements.
    gamma = mx.mean(mx.abs(w))
    # An all-zero matrix gets scale 1, so that nothing divides by zero.
    scale = mx.where(gamma > 0, gamma, mx.ones_like(gamma))
    # round() rounds half to even, so +-0.5 become 0; the clip folds +-2, +-3, ... to +-1.
    t = mx.clip(mx.round(w / scale), -1, 1)
    return t.astype(mx.int8), scale


def ternary_matmul(x: mx.array, t: mx.array, scale: mx.array) -> mx.array:
    # (..., 1, in) against t of shape (out, in): every output row selects its inputs.
    xe = mx.expand_dims(x, -2)
    zero = mx.zeros_like(xe)
    # Compare with 1 and -1 explicitly: inputs whose weight is 0 are in neither sum.
    plus = mx.where(t == 1, xe, zero).sum(axis=-1)
    minus = mx.where(t == -1, xe, zero).sum(axis=-1)
    # The only multiplication of the layer: one per output.
    return scale * (plus - minus)
