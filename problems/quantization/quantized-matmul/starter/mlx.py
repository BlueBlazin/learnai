import mlx.core as mx


def qmatmul(
    x: mx.array, codes: mx.array, scale: mx.array, bias: mx.array, group_size: int
) -> mx.array:
    """x @ W_hat.T for the group-quantized matrix W_hat = codes * scale + bias, without building it.

    x: float32, shape (..., in), one to three dimensions.
    codes: uint8, shape (out, in). scale, bias: float32, shape (out, in // group_size);
    entry [i, g] belongs to inputs g * group_size ... (g + 1) * group_size - 1 of row i.
    Returns a float32 tensor of shape (..., out).
    """
    raise NotImplementedError
