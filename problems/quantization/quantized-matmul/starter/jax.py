import jax
import jax.numpy as jnp


def qmatmul(
    x: jax.Array, codes: jax.Array, scale: jax.Array, bias: jax.Array, group_size: int
) -> jax.Array:
    """x @ W_hat.T for the group-quantized matrix W_hat = codes * scale + bias, without building it.

    x: float32, shape (..., in), one to three dimensions.
    codes: uint8, shape (out, in). scale, bias: float32, shape (out, in // group_size);
    entry [i, g] belongs to inputs g * group_size ... (g + 1) * group_size - 1 of row i.
    Returns a float32 tensor of shape (..., out).
    """
    raise NotImplementedError
