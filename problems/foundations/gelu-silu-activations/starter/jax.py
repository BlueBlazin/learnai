import math

import jax
import jax.numpy as jnp


def gelu(x: jax.Array, approximate: str = "none") -> jax.Array:
    """GELU of `x`, elementwise: the exact erf form or the tanh approximation.

    x: float32 array of any shape; every entry is finite with |x| <= 1e4.
    approximate: "none" for x * Phi(x) computed with erf, "tanh" for
        0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3))).
    Returns a float32 array of the same shape as x.
    """
    raise NotImplementedError


def silu(x: jax.Array) -> jax.Array:
    """SiLU (swish) of `x`, elementwise: x * sigmoid(x), with no NaN for large |x|.

    x: float32 array of any shape; every entry is finite with |x| <= 1e4.
    Returns a float32 array of the same shape as x.
    """
    raise NotImplementedError
