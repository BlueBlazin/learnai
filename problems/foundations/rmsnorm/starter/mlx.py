import mlx.core as mx


def rms_norm(x: mx.array, weight: mx.array, eps: float = 1e-5) -> mx.array:
    """RMSNorm over the last axis: x / sqrt(mean(x^2) + eps) * weight.

    x: float32 array of shape (..., d), 1 to 4 dimensions; |x| may be as large as 1e30.
    weight: float32 array of shape (d,), the learned gain.
    eps: positive float, added to the mean square inside the square root.
    Returns a float32 array of the same shape as x.
    """
    raise NotImplementedError
