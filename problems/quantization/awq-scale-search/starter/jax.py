import jax


def awq_scales(
    w: jax.Array, x_calib: jax.Array, bits: int, group_size: int, grid: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Search the per-input-channel scale that minimizes the quantized layer's output error.

    w: float32, shape (out, in), the weight of a linear layer; in divisible by group_size.
    x_calib: float32, shape (N, in), calibration inputs of the layer.
    bits: 2 to 8. group_size: at least 1. grid: float32, shape (G,), exponents in [0, 1].
    Returns (scales, alpha): the float32 scales of shape (in,) for the best exponent,
    and that exponent as a 0-d float32 tensor (one of the values of `grid`).
    """
    raise NotImplementedError
