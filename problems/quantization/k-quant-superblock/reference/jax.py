import jax
import jax.numpy as jnp

BLOCKS = 8  # blocks per superblock
BLOCK = 32  # weights per block
CODE_MAX = 15  # 4-bit codes
FIELD_MAX = 63  # 6-bit block scales and minimums


def _six_bit(value: jax.Array, largest: jax.Array) -> jax.Array:
    # value as a multiple of largest / 63, rounded: an integer in 0..63. A superblock
    # whose largest value is 0 has nothing to represent; the where keeps 0 / 0 away.
    safe = jnp.where(largest > 0, largest, 1.0)
    return jnp.where(largest > 0, jnp.round(FIELD_MAX * value / safe), 0.0)


def q4k_encode(
    w: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    n = w.shape[0]
    blocks = w.reshape(n, BLOCKS, BLOCK)
    # Level 1: a 4-bit grid per block. The format stores a non-negative offset that is
    # subtracted, so the bottom of a block's grid can be 0 or negative, never positive.
    lo = jnp.minimum(jnp.min(blocks, axis=-1), 0.0)
    hi = jnp.max(blocks, axis=-1)
    block_scale = (hi - lo) / CODE_MAX
    block_min = -lo
    # Level 2: the eight scales and the eight minimums are themselves quantized to 6
    # bits, against the largest of each in the superblock.
    max_scale = jnp.max(block_scale, axis=-1, keepdims=True)
    max_min = jnp.max(block_min, axis=-1, keepdims=True)
    scales = _six_bit(block_scale, max_scale)
    mins = _six_bit(block_min, max_min)
    d = (max_scale / FIELD_MAX).astype(jnp.float16)
    dmin = (max_min / FIELD_MAX).astype(jnp.float16)
    # The codes are computed against the step and offset a decoder will reconstruct
    # (float16 units times 6-bit integers), not against the ideal ones of level 1.
    step = d.astype(jnp.float32) * scales
    offset = dmin.astype(jnp.float32) * mins
    safe = jnp.where(step > 0, step, 1.0)
    codes = jnp.clip(jnp.round((blocks + offset[..., None]) / safe[..., None]), 0, CODE_MAX)
    codes = jnp.where(step[..., None] > 0, codes, 0.0)
    return (
        codes.reshape(n, BLOCKS * BLOCK).astype(jnp.uint8),
        scales.astype(jnp.uint8),
        mins.astype(jnp.uint8),
        d[:, 0],
        dmin[:, 0],
    )


def q4k_decode(
    codes: jax.Array,
    scales: jax.Array,
    mins: jax.Array,
    d: jax.Array,
    dmin: jax.Array,
) -> jax.Array:
    n = codes.shape[0]
    # Widen everything to float32 first: 63 * 15 does not fit in a byte.
    step = d.astype(jnp.float32)[:, None] * scales.astype(jnp.float32)
    offset = dmin.astype(jnp.float32)[:, None] * mins.astype(jnp.float32)
    q = codes.astype(jnp.float32).reshape(n, BLOCKS, BLOCK)
    return (step[..., None] * q - offset[..., None]).reshape(n, BLOCKS * BLOCK)
