import jax


def inverse_hessian_factor(x_calib: jax.Array, percdamp: float) -> jax.Array:
    """The upper Cholesky factor U of the dampened inverse Hessian, H^-1 = U^T U.

    x_calib: float32, shape (N, in), calibration inputs of the layer.
    percdamp: the dampening, as a fraction of the mean diagonal of H.
    Returns a float32 upper-triangular tensor of shape (in, in).
    """
    raise NotImplementedError


def gptq_quantize(w: jax.Array, u: jax.Array, bits: int) -> tuple[jax.Array, jax.Array]:
    """Quantize w column by column, propagating each column's error to the later columns.

    w: float32, shape (out, in). u: float32, shape (in, in), as inverse_hessian_factor
    returns it. bits: 2 to 8.
    Returns (codes, scale): int8 codes of shape (out, in) in [-qmax, qmax], and the
    float32 per-row scale of shape (out,), so that the quantized weight is
    codes * scale[:, None].
    """
    raise NotImplementedError
