import mlx.core as mx


def normalize_over(x: mx.array, dim: int) -> mx.array:
    # keepdims leaves the summed axis in place with size 1, so the sum lines up with x
    # along every other axis whatever dim is.
    return x / mx.sum(x, axis=dim, keepdims=True)


def last_step(x: mx.array) -> mx.array:
    # An integer index removes exactly the axis it indexes: (B, T, D) -> (B, D).
    return x[:, -1]


def column(x: mx.array, j: int) -> mx.array:
    # A slice of length 1 keeps the axis: (B, V) -> (B, 1).
    return x[:, j : j + 1]
