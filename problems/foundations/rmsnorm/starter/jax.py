import jax
import jax.numpy as jnp


def rms_norm(x: jax.Array, weight: jax.Array, eps: float = 1e-5) -> jax.Array:
    """RMSNorm over the last axis: x / sqrt(mean(x^2) + eps) * weight.

    x: float32 array of shape (..., d), 1 to 4 dimensions; |x| may be as large as 1e30.
    weight: float32 array of shape (d,), the learned gain.
    eps: positive float, added to the mean square inside the square root.
    Returns a float32 array of the same shape as x.
    """
    raise NotImplementedError
