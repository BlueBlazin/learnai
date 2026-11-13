import mlx.core as mx


def normalize(x, axis=-1):
    return x / mx.sqrt(mx.sum(x * x, axis=axis, keepdims=True))
