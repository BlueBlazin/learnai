import jax.numpy as jnp


def normalize(x, axis=-1):
    return x / jnp.sqrt(jnp.sum(x * x, axis=axis, keepdims=True))
