import jax
import jax.numpy as jnp
from jax.scipy.special import ndtri

OFFSET = 0.9677083  # the outermost quantile; the inverse CDF there is about 1.848


def nf4_levels() -> jax.Array:
    # Eight positive levels and seven negative ones: evenly spaced probabilities from
    # OFFSET down to (but excluding) 0.5, pushed through the inverse normal CDF.
    positive = ndtri(jnp.linspace(OFFSET, 0.5, 9)[:-1])
    negative = -ndtri(jnp.linspace(OFFSET, 0.5, 8)[:-1])
    # Ascending order, with an exact 0 between the two halves.
    levels = jnp.concatenate([negative, jnp.zeros(1), positive[::-1]])
    # Both ends come from the same quantile, so this maps them to -1 and 1.
    return levels / jnp.max(levels)


def nf4_quantize(w: jax.Array, block_size: int) -> tuple[jax.Array, jax.Array]:
    out_dim, in_dim = w.shape
    # Consecutive runs of block_size inputs: (out, n_blocks, block_size).
    blocks = w.reshape(out_dim, in_dim // block_size, block_size)
    absmax = jnp.max(jnp.abs(blocks), axis=-1)
    # An all-zero block gets scale 1, so that the division below is 0 / 1.
    absmax = jnp.where(absmax > 0, absmax, jnp.ones_like(absmax))
    normalized = blocks / absmax[..., None]  # in [-1, 1]
    # The nearest level by value: compare with all 16 and take the closest.
    codes = jnp.argmin(jnp.abs(normalized[..., None] - nf4_levels()), axis=-1)
    return codes.reshape(out_dim, in_dim).astype(jnp.uint8), absmax


def nf4_dequantize(codes: jax.Array, absmax: jax.Array, block_size: int) -> jax.Array:
    # A table lookup, then each block is scaled back by its own absmax.
    values = nf4_levels()[codes]
    return values * jnp.repeat(absmax, block_size, axis=1)
