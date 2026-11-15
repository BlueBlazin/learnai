import jax
import jax.numpy as jnp

QMIN, QMAX = -128, 127


def int8_linear(
    x: jax.Array, w_q: jax.Array, w_scale: jax.Array, bias: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """A linear layer on int8 weights and dynamically quantized int8 activations.

    x: float32 array of shape (n, in); each row is one token and gets its own affine
    int8 grid (scale and zero-point from the row's range, widened to include 0).
    w_q: int8 array of shape (out, in), codes in [-127, 127]. w_scale: float32 array
    of shape (out,), one scale per row of w_q. bias: float32 array of shape (out,).
    Returns (y, acc): y is a float32 array of shape (n, out); acc is the int32 array
    of shape (n, out) holding the raw integer products x_q @ w_q.T.
    """
    raise NotImplementedError
