import jax
import jax.numpy as jnp

# How many bits are set in each possible byte: popcount by table lookup.
POPCOUNT = jnp.array([byte.bit_count() for byte in range(256)])
# The value of each of the eight bits of a byte, the first bit most significant.
BIT_VALUES = jnp.array([128, 64, 32, 16, 8, 4, 2, 1])


def binarize(e: jax.Array) -> jax.Array:
    bits = (e > 0).reshape(*e.shape[:-1], -1, 8)  # (..., D / 8, 8): one row per byte
    return (bits * BIT_VALUES).sum(axis=-1).astype(jnp.uint8)


def hamming_topk(q_bits: jax.Array, corpus_bits: jax.Array, m: int) -> tuple[jax.Array, jax.Array]:
    # XOR leaves a 1 wherever two codes differ; the distance is the number of 1 bits.
    differing = jnp.bitwise_xor(corpus_bits, q_bits)  # (N, D / 8) uint8
    distances = POPCOUNT[differing].sum(axis=-1)  # (N,)
    # jnp.argsort is stable: equal distances are ordered by index.
    order = jnp.argsort(distances)[:m]
    return order, distances[order]


def search(
    q: jax.Array,
    corpus_bits: jax.Array,
    corpus_int8: jax.Array,
    k: int,
    oversample: int,
    distance,
) -> tuple[jax.Array, jax.Array]:
    # Stage 1: cheap and coarse. More candidates than needed, by Hamming distance.
    candidates, _ = hamming_topk(binarize(q), corpus_bits, k * oversample)
    # Stage 2: precise, on the candidates only. The query keeps its full precision.
    scores = distance(q, corpus_int8[candidates])
    order = jnp.argsort(-scores)[:k]  # stable, descending
    return candidates[order], scores[order]
