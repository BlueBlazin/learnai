import mlx.core as mx

# Bit offset of each of the eight codes in a word: code i sits at bits 4i to 4i + 3.
SHIFTS = 4 * mx.arange(8, dtype=mx.uint32)


def pack4(codes: mx.array) -> mx.array:
    # Widen before shifting: a uint8 shifted left stays uint8 and loses its high bits.
    padded = mx.pad(codes.astype(mx.uint32), (0, -codes.shape[0] % 8))
    # The eight shifted codes occupy disjoint bits, so adding them is the same as OR.
    return mx.sum(padded.reshape(-1, 8) << SHIFTS, axis=1).astype(mx.uint32)


def unpack4(words: mx.array, n: int) -> mx.array:
    # The words are unsigned, so the right shift brings in zeros, never sign bits.
    nibbles = (words.astype(mx.uint32)[:, None] >> SHIFTS) & 0xF
    # Row-major flattening restores the order; the padding codes are at the end.
    return nibbles.reshape(-1)[:n].astype(mx.uint8)
