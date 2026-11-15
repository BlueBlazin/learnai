import mlx.core as mx

# How many bits are set in each possible byte: popcount by table lookup.
POPCOUNT = mx.array([byte.bit_count() for byte in range(256)])
# The value of each of the eight bits of a byte, the first bit most significant.
BIT_VALUES = mx.array([128, 64, 32, 16, 8, 4, 2, 1])


def binarize(e: mx.array) -> mx.array:
    bits = (e > 0).reshape(*e.shape[:-1], -1, 8)  # (..., D / 8, 8): one row per byte
    return (bits * BIT_VALUES).sum(axis=-1).astype(mx.uint8)


def hamming_topk(q_bits: mx.array, corpus_bits: mx.array, m: int) -> tuple[mx.array, mx.array]:
    # XOR leaves a 1 wherever two codes differ; the distance is the number of 1 bits.
    differing = mx.bitwise_xor(corpus_bits, q_bits)  # (N, D / 8) uint8
    distances = POPCOUNT[differing].sum(axis=-1)  # (N,)
    # mx.argsort is stable: equal distances are ordered by index.
    order = mx.argsort(distances)[:m]
    return order, distances[order]


def search(
    q: mx.array,
    corpus_bits: mx.array,
    corpus_int8: mx.array,
    k: int,
    oversample: int,
    distance,
) -> tuple[mx.array, mx.array]:
    # Stage 1: cheap and coarse. More candidates than needed, by Hamming distance.
    candidates, _ = hamming_topk(binarize(q), corpus_bits, k * oversample)
    # Stage 2: precise, on the candidates only. The query keeps its full precision.
    scores = distance(q, corpus_int8[candidates])
    order = mx.argsort(-scores)[:k]  # stable, descending
    return candidates[order], scores[order]
