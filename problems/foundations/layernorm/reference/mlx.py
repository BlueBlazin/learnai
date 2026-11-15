import mlx.core as mx


def layer_norm(x: mx.array, weight: mx.array, bias: mx.array, eps: float = 1e-5) -> mx.array:
    # Center each vector, then divide by its standard deviation. The variance is the
    # mean of the squared deviations (divided by d, not d - 1), with eps added inside
    # the square root.
    centered = x - mx.mean(x, axis=-1, keepdims=True)
    var = mx.mean(mx.square(centered), axis=-1, keepdims=True)
    return centered / mx.sqrt(var + eps) * weight + bias
