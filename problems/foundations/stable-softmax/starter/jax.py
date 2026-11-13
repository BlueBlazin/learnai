import jax
import jax.numpy as jnp


def softmax(x: jax.Array, axis: int = -1) -> jax.Array:
    """Softmax of `x` along `axis`, correct for any float32 logits.

    x: float32 array with 1 to 4 dimensions; entries may be -inf (masked).
    Returns a float32 array of the same shape whose slices along `axis` sum to 1.
    """
    raise NotImplementedError


def log_softmax(x: jax.Array, axis: int = -1) -> jax.Array:
    """Log of the softmax of `x` along `axis`, without underflowing to -inf.

    x: float32 array with 1 to 4 dimensions; entries may be -inf (masked).
    Returns a float32 array of the same shape.
    """
    raise NotImplementedError
