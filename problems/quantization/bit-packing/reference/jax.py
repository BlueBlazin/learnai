import jax
import jax.numpy as jnp

# Bit offset of each of the eight codes in a word: code i sits at bits 4i to 4i + 3.
SHIFTS = 4 * jnp.arange(8, dtype=jnp.uint32)


def pack4(codes: jax.Array) -> jax.Array:
    # Widen before shifting: a uint8 shifted left stays uint8 and loses its high bits.
    padded = jnp.pad(codes.astype(jnp.uint32), (0, -codes.shape[0] % 8))
    # The eight shifted codes occupy disjoint bits, so adding them is the same as OR.
    return jnp.sum(padded.reshape(-1, 8) << SHIFTS, axis=1, dtype=jnp.uint32)


def unpack4(words: jax.Array, n: int) -> jax.Array:
    # The words are unsigned, so the right shift brings in zeros, never sign bits.
    nibbles = (words.astype(jnp.uint32)[:, None] >> SHIFTS) & 0xF
    # Row-major flattening restores the order; the padding codes are at the end.
    return nibbles.reshape(-1)[:n].astype(jnp.uint8)
