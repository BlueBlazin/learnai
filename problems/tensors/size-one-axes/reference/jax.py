import jax
import jax.numpy as jnp


def normalize_over(x: jax.Array, dim: int) -> jax.Array:
    # keepdims leaves the summed axis in place with size 1, so the sum lines up with x
    # along every other axis whatever dim is.
    return x / jnp.sum(x, axis=dim, keepdims=True)


def last_step(x: jax.Array) -> jax.Array:
    # An integer index removes exactly the axis it indexes: (B, T, D) -> (B, D).
    return x[:, -1]


def column(x: jax.Array, j: int) -> jax.Array:
    # A slice of length 1 keeps the axis: (B, V) -> (B, 1).
    return x[:, j : j + 1]
