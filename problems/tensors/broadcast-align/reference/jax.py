import jax
import jax.numpy as jnp


def relative_positions(T: int) -> jax.Array:
    pos = jnp.arange(T)
    # A column of i against a row of j: (T, 1) - (1, T) broadcasts to (T, T).
    return pos[:, None] - pos[None, :]


def scale_rows(x: jax.Array, s: jax.Array) -> jax.Array:
    # (B,) lines up with the last axis of x. (B, 1, 1) lines up with (B, T, D).
    return x * s[:, None, None]


def pairwise_sq_dist(a: jax.Array, b: jax.Array) -> jax.Array:
    # (N, 1, D) - (1, M, D) broadcasts to (N, M, D): every point of a against every
    # point of b. The feature axis is then summed away.
    diff = a[:, None, :] - b[None, :, :]
    return jnp.sum(diff * diff, axis=-1)
