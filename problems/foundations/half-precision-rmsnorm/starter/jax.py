import math

import jax
import jax.numpy as jnp


def rms_norm_f16(x: jax.Array, weight: jax.Array, eps: float = 1e-5) -> jax.Array:
    """RMSNorm over the last axis, x / sqrt(mean(x^2) + eps) * weight, entirely in float16.

    x: float16 array of shape (..., d), 1 to 4 dimensions; any finite float16 values,
        from subnormals (6e-8) up to the largest float16 (65504).
    weight: float16 array of shape (d,), the learned gain.
    eps: positive Python float, added to the mean square inside the square root.
    Returns a float16 array of the same shape as x. Every floating-point array
    computed on the way must be float16 as well.
    """
    raise NotImplementedError
