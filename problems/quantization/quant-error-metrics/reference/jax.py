import jax
import jax.numpy as jnp


def mse(a: jax.Array, b: jax.Array) -> jax.Array:
    # One number for the whole tensor: the mean runs over every element.
    return jnp.mean((a - b) ** 2)


def sqnr_db(ref: jax.Array, approx: jax.Array) -> jax.Array:
    signal = jnp.sum(ref**2)
    noise = jnp.sum((ref - approx) ** 2)
    # A ratio of powers (squared quantities) takes 10 log10. Zero noise gives +inf.
    return 10.0 * jnp.log10(signal / noise)


def max_abs_error(a: jax.Array, b: jax.Array) -> jax.Array:
    return jnp.max(jnp.abs(a - b))


def output_error(w: jax.Array, w_hat: jax.Array, x: jax.Array) -> jax.Array:
    # A linear layer computes x w^T: one row of outputs per row of x.
    y = x @ w.T
    y_hat = x @ w_hat.T
    # Frobenius norms over the whole output, normalized by the exact output.
    return jnp.sqrt(jnp.sum((y - y_hat) ** 2)) / jnp.sqrt(jnp.sum(y**2))
