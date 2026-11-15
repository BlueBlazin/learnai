import jax
import jax.numpy as jnp


def layer_norm(x: jax.Array, weight: jax.Array, bias: jax.Array, eps: float = 1e-5) -> jax.Array:
    # Center each vector, then divide by its standard deviation. The variance is the
    # mean of the squared deviations (divided by d, not d - 1), with eps added inside
    # the square root.
    centered = x - jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean(jnp.square(centered), axis=-1, keepdims=True)
    return centered / jnp.sqrt(var + eps) * weight + bias
