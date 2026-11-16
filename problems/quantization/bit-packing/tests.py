"""Tests for bit-packing. Words and codes are integers: every comparison is exact."""

import numpy as np

from learnai.judge import case, check

# One row of 64 weights, np.linspace(-1, 1, 64), quantized by MLX 0.32.4 with
# mx.quantize(w, group_size=64, bits=4): the packed words it returned, and the 4-bit
# codes round((w - bias) / scale) they hold.
MLX_WORDS = [
    0xCDDDDEEE, 0xBBBBCCCC, 0x999AAAAA, 0x77888889,
    0x56666677, 0x44444555, 0x22223333, 0x00011112,
]  # fmt: skip
MLX_CODES = [
    14, 14, 14, 13, 13, 13, 13, 12, 12, 12, 12, 12, 11, 11, 11, 11,
    10, 10, 10, 10, 10, 9, 9, 9, 9, 8, 8, 8, 8, 8, 7, 7,
    7, 7, 6, 6, 6, 6, 6, 5, 5, 5, 5, 4, 4, 4, 4, 4,
    3, 3, 3, 3, 2, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0,
]  # fmt: skip


def u8(values):
    return np.array(values, dtype=np.uint8)


def u32(values):
    return np.array(values, dtype=np.uint32)


def hex_words(words):
    return "[" + ", ".join(f"0x{int(w):08X}" for w in np.asarray(words).ravel()[:12]) + "]"


def check_pack(ctx, codes, want=None):
    """pack4(codes) must be a uint32 vector equal to the expected words."""
    got = ctx.call("pack4", codes)
    want = u32(ctx.expected("pack4", codes) if want is None else want)
    check.shape(got, want.shape, label="pack4")
    check.that(
        got.dtype == np.uint32, "pack4 must return a uint32 tensor", f"got dtype {got.dtype}"
    )
    wrong = int(np.count_nonzero(got != want))
    check.that(
        wrong == 0,
        f"pack4: {wrong} of {want.size} words are wrong",
        f"codes = {codes[:24].tolist()}{' ...' if codes.size > 24 else ''}\n"
        f"expected (first words) = {hex_words(want)}\n"
        f"got      (first words) = {hex_words(got)}",
    )
    return got


def check_unpack(ctx, words, n, want=None):
    """unpack4(words, n) must be a uint8 vector of n codes equal to the expected ones."""
    got = ctx.call("unpack4", words, n)
    want = u8(ctx.expected("unpack4", words, n) if want is None else want)
    check.shape(got, (n,), label="unpack4")
    check.that(
        got.dtype == np.uint8, "unpack4 must return a uint8 tensor", f"got dtype {got.dtype}"
    )
    check.equal(got, want, label=f"unpack4(words, {n})")
    return got


def check_both(ctx, codes):
    """pack4 against the oracle, unpack4 against the oracle, then the round trip."""
    n = int(codes.size)
    words = check_pack(ctx, codes)
    check_unpack(ctx, u32(ctx.expected("pack4", codes)), n, want=codes)
    back = ctx.call("unpack4", words, n)
    check.equal(back, codes, label="unpack4(pack4(codes), n), the round trip")


@case("the worked example: 11 codes become two words", public=True)
def worked_example(ctx):
    codes = u8([3, 14, 0, 9, 5, 12, 1, 6, 15, 2, 8])
    check_pack(ctx, codes, [0x61C590E3, 0x0000082F])
    check_unpack(ctx, u32([0x61C590E3, 0x0000082F]), 11, codes)
    check_both(ctx, codes)


# Rejects signed_shift: the top code of several words is 8 or more.
@case("a row of 64 weights exactly as mx.quantize packed it", public=True)
def mlx_layout(ctx):
    check_unpack(ctx, u32(MLX_WORDS), 64, MLX_CODES)
    check_pack(ctx, u8(MLX_CODES), MLX_WORDS)


# Rejects high_nibble_first, swaps_nibbles_in_byte, and no_widening one word at a time.
@case("one full word: every position, and codes that set the top bit", public=True)
def single_words(ctx):
    check_pack(ctx, u8([0, 1, 2, 3, 4, 5, 6, 7]), [0x76543210])
    check_pack(ctx, u8([8, 9, 10, 11, 12, 13, 14, 15]), [0xFEDCBA98])
    check_pack(ctx, u8([15] * 8), [0xFFFFFFFF])
    check_pack(ctx, u8([0] * 8), [0x00000000])
    for position in range(8):
        codes = np.zeros(8, dtype=np.uint8)
        codes[position] = 0xF
        check_both(ctx, codes)
    check_unpack(ctx, u32([0xFEDCBA98, 0x80000000]), 16)


# Rejects returns_padding and drops_partial_word.
@case("lengths that are not a multiple of 8 are padded with zero codes", public=True)
def padding(ctx):
    for n in (1, 7, 9, 20):
        codes = ctx.rng.integers(1, 16, size=n).astype(np.uint8)
        words = check_pack(ctx, codes)
        check.shape(words, ((n + 7) // 8,), label="pack4")
        check_both(ctx, codes)


@case("unpack4 returns the first n codes, whatever the remaining bits hold")
def unpack_prefix(ctx):
    words = ctx.rng.integers(0, 2**32, size=5, dtype=np.uint64).astype(np.uint32)
    for n in (40, 37, 16, 1):
        check_unpack(ctx, words, n)


@case("unpacking words whose top bit is set")
def top_bit_set(ctx):
    words = u32([0x80000000, 0xF0000000, 0xFFFFFFFF, 0x90000001, 0x7FFFFFFF, 0xA5A5A5A5])
    check_unpack(ctx, words, 48)
    check_unpack(ctx, words, 45)
    check_both(ctx, u8([0, 0, 0, 0, 0, 0, 0, 15, 1, 2, 3, 4, 5, 6, 7, 8, 9]))


@case("all sixteen code values in every position")
def every_value(ctx):
    codes = np.stack([np.roll(np.arange(16, dtype=np.uint8), k) for k in range(8)], axis=1)
    check_both(ctx, codes.reshape(-1))


@case("a long vector: 4,096 codes, and 4,099 codes")
def long_vectors(ctx):
    for n in (4096, 4099):
        check_both(ctx, ctx.rng.integers(0, 16, size=n).astype(np.uint8))


@case("round trip: unpack4(pack4(codes), n) returns codes for random lengths")
def round_trip(ctx):
    def prop(rng):
        n = int(rng.integers(1, 200))
        codes = rng.integers(0, 16, size=n).astype(np.uint8)
        words = ctx.call("pack4", codes)
        check.shape(words, ((n + 7) // 8,), label="pack4")
        back = ctx.call("unpack4", words, n)
        check.shape(back, (n,), label="unpack4(pack4(codes), n)")
        check.equal(back, codes, label="unpack4(pack4(codes), n)")
        # The words themselves must also be the specified layout, padding included.
        check_pack(ctx, codes)

    check.forall(prop, trials=5)
