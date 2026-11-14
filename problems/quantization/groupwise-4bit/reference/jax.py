import jax
import jax.numpy as jnp


def quantize_groups(
    w: jax.Array, bits: int, group_size: int
) -> tuple[jax.Array, jax.Array, jax.Array]:
    out_dim, in_dim = w.shape
    levels = 2**bits - 1  # the largest code: 2**bits codes have 2**bits - 1 steps between them
    # Split every row into consecutive runs of group_size inputs: (out, n_groups, group_size).
    groups = w.reshape(out_dim, in_dim // group_size, group_size)
    lo = jnp.min(groups, axis=-1)
    hi = jnp.max(groups, axis=-1)
    # A constant group has hi == lo: give it scale 1, so that the division below is 0 / 1.
    scale = jnp.where(hi > lo, (hi - lo) / levels, jnp.ones_like(hi))
    codes = jnp.round((groups - lo[..., None]) / scale[..., None])
    codes = jnp.clip(codes, 0, levels).astype(jnp.uint8)
    return codes.reshape(out_dim, in_dim), scale, lo
