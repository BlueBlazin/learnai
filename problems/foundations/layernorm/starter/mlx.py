import mlx.core as mx


def layer_norm(x: mx.array, weight: mx.array, bias: mx.array, eps: float = 1e-5) -> mx.array:
    """LayerNorm over the last axis: (x - mean) / sqrt(var + eps) * weight + bias.

    x: float32 array of shape (..., d), 1 to 4 dimensions.
    weight: float32 array of shape (d,), the learned gain.
    bias: float32 array of shape (d,), the learned bias.
    eps: positive float, added to the variance inside the square root.
    The variance is the biased one: squared deviations from the mean, divided by d.
    Returns a float32 array of the same shape as x.
    """
    raise NotImplementedError
