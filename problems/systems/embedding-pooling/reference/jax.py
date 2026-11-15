import jax
import jax.numpy as jnp

EPS = 1e-12


def _normalize(x: jax.Array) -> jax.Array:
    # Divide by the L2 norm, but never by less than EPS: a zero vector stays zero
    # instead of becoming 0 / 0.
    return x / jnp.maximum(jnp.linalg.norm(x, axis=-1, keepdims=True), EPS)


def pool(states: jax.Array, mask: jax.Array, mode: str) -> jax.Array:
    batch, length, _ = states.shape
    positions = jnp.arange(length)
    if mode == "mean":
        # Zero the padded states, then divide by the number of real tokens, not by T.
        weights = mask.astype(states.dtype)[..., None]  # (B, T, 1)
        pooled = (states * weights).sum(axis=1) / weights.sum(axis=1)
    else:
        if mode == "cls":
            # The first real token: the smallest position whose mask is set.
            index = jnp.where(mask != 0, positions, length).min(axis=1)
        else:
            # The last real token: the largest position whose mask is set. With right
            # padding that is lengths - 1; with left padding it is T - 1.
            index = (positions * (mask != 0)).max(axis=1)
        pooled = states[jnp.arange(batch), index]  # (B, D)
    return _normalize(pooled)


def truncate_matryoshka(e: jax.Array, dim: int) -> jax.Array:
    # A prefix of a unit vector is shorter than 1: slice, then normalize again.
    return _normalize(e[..., :dim])
