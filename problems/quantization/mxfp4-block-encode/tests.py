"""Tests for mxfp4-block-encode.

Everything here is exact. The shared scale is a power of two, so x / X has no rounding
error, and the nearest FP4 value (with ties to the even code) is a matter of
arithmetic, not of float32 noise. The expected codes and scale bytes are therefore
computed by `model_encode` below, in float64 and by brute force over the eight
magnitudes, independently of any reference; `np.frexp` supplies the exact exponent.

JAX compiles every new shape, so the cases share three layouts: (1, 64), (4, 64), and
(32, 32).
"""

import numpy as np

from whetstone.judge import case, check

BLOCK = 32
MAGNITUDES = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
# Every value halfway between two adjacent magnitudes, and the code it must round to.
TIES = {0.25: 0, 0.75: 2, 1.25: 2, 1.75: 4, 2.5: 4, 3.5: 6, 5.0: 6}


def model_encode(x):
    """(codes, exps) of the statement, by brute force in float64."""
    out_dim, in_dim = x.shape
    blocks = x.reshape(out_dim, in_dim // BLOCK, BLOCK).astype(np.float64)
    largest = np.abs(blocks).max(axis=-1)
    _, exponent = np.frexp(largest)  # largest = mantissa * 2**exponent, mantissa in [0.5, 1)
    shared = np.where(largest > 0, exponent - 1 - 2, 0)
    a = np.minimum(np.abs(blocks) / (2.0**shared)[..., None], 6.0)
    distance = np.abs(a[..., None] - MAGNITUDES)
    nearest = distance == distance.min(axis=-1, keepdims=True)
    even = nearest & (np.arange(8) % 2 == 0)
    # At most two candidates, adjacent codes: prefer the even one.
    magnitude = np.where(even.any(axis=-1), even.argmax(axis=-1), nearest.argmax(axis=-1))
    codes = magnitude + 8 * (blocks < 0)
    return codes.reshape(out_dim, in_dim).astype(np.uint8), (shared + 127).astype(np.uint8)


def model_decode(codes, exps):
    """The values the codes stand for, in float64."""
    c = codes.astype(np.int64)
    scale = 2.0 ** (exps.astype(np.float64) - 127)
    return (1 - 2 * (c >> 3)) * MAGNITUDES[c & 7] * np.repeat(scale, BLOCK, axis=1)


def check_encode(ctx, x):
    """mxfp4_encode against the model and the oracle; dtype and range of both outputs."""
    x = np.asarray(x, dtype=np.float32)
    codes, exps = ctx.call("mxfp4_encode", x)
    want_codes, want_exps = model_encode(x)
    oracle_codes, oracle_exps = ctx.expected("mxfp4_encode", x)
    assert np.array_equal(oracle_codes, want_codes) and np.array_equal(oracle_exps, want_exps)
    check.shape(exps, want_exps.shape, label="exps")
    check.that(exps.dtype == np.uint8, "exps must be a uint8 tensor", f"dtype {exps.dtype}")
    check.equal(exps, want_exps, label="exps")
    check.shape(codes, x.shape, label="codes")
    check.that(codes.dtype == np.uint8, "codes must be a uint8 tensor", f"dtype {codes.dtype}")
    check.equal(codes, want_codes, label="codes")
    return codes, exps


def check_decode(ctx, codes, exps):
    """mxfp4_decode against the model and the oracle."""
    values = ctx.check("mxfp4_decode", codes, exps, atol=1e-30)
    check.allclose(values, model_decode(codes, exps), atol=1e-30, label="mxfp4_decode")
    return values


def check_round_trip(ctx, x):
    """Encode, then decode the solution's own codes."""
    codes, exps = check_encode(ctx, x)
    return codes, exps, check_decode(ctx, codes, exps)


def scaled_blocks(ctx, out_dim, n_blocks, *, low=-12.0, high=12.0):
    """Normal values whose blocks each have a magnitude of their own, 2**low to 2**high."""
    magnitude = 2.0 ** ctx.uniform(out_dim, n_blocks, 1, low=low, high=high)
    x = ctx.randn(out_dim, n_blocks, BLOCK) * magnitude
    return x.reshape(out_dim, n_blocks * BLOCK).astype(np.float32)


def worked_input():
    x = np.zeros((1, 64), dtype=np.float32)
    x[0, 0:8] = [5.0, -3.2, 0.3, 0.75, 1.0, -0.2, 2.5, 7.0]
    x[0, 32:36] = [0.11, -0.05, 0.02, 0.004]
    return x


@case("the worked example: one row, two blocks", public=True)
def worked_example(ctx):
    x = worked_input()
    codes, exps = ctx.call("mxfp4_encode", x)
    check.equal(exps, np.array([[127, 121]]), label="exps")
    check.equal(codes[0, 0:8], np.array([6, 13, 1, 2, 2, 8, 4, 7]), label="codes[0, 0:8]")
    check.equal(codes[0, 32:36], np.array([7, 13, 3, 1]), label="codes[0, 32:36]")
    _, _, values = check_round_trip(ctx, x)
    check.allclose(
        values[0, 0:8],
        np.array([4.0, -3.0, 0.5, 1.0, 1.0, 0.0, 2.0, 6.0], np.float32),
        label="decoded block 0",
    )
    check.allclose(
        values[0, 32:36],
        np.array([0.09375, -0.046875, 0.0234375, 0.0078125], np.float32),
        label="decoded block 1",
    )


# Rejects per_element_exponent, round_toward_zero, forgets_emax, whole_row_exponent.
@case("a (4, 64) matrix whose blocks are on different scales", public=True)
def blocks_on_different_scales(ctx):
    check_round_trip(ctx, scaled_blocks(ctx, 4, 2))


# Rejects ties_away_from_zero, and any scale that is not an exact power of two.
@case("values exactly halfway between two FP4 values go to the even code", public=True)
def exact_ties(ctx):
    ties = np.array(list(TIES), dtype=np.float32)
    want = np.array(list(TIES.values()))
    x = np.zeros((4, 64), dtype=np.float32)
    powers = [[1.0, 2.0**-7], [2.0**5, 2.0**-20], [2.0**13, 2.0**-1], [2.0**-11, 2.0**30]]
    for row in range(4):
        for block in range(2):
            start, p = 32 * block, np.float32(powers[row][block])
            x[row, start] = 4.0 * p  # the block maximum: the scale is exactly p
            x[row, start + 1 : start + 8] = ties * p
            x[row, start + 8 : start + 15] = -ties * p
    codes, _ = check_encode(ctx, x)
    check.equal(codes[:, 1:8], np.tile(want, (4, 1)), label="codes of the positive ties")
    check.equal(codes[:, 8:15], np.tile(want + 8, (4, 1)), label="codes of the negative ties")


# Rejects no_saturation and exponent_avoids_saturation.
@case("a block maximum above 6 scales saturates at the largest element value", public=True)
def saturation(ctx):
    x = np.zeros((4, 64), dtype=np.float32)
    maxima = [7.0, -7.9, 6.5, 6.0, -7.0, 112.0, 0.21, -1.9e-3]
    for i, value in enumerate(maxima):
        row, start = divmod(i, 2)
        x[row, 32 * start] = value
        x[row, 32 * start + 1 : 32 * start + 32] = ctx.randn(31) * abs(value) / 4
    np.clip(x, -np.abs(x[:, ::32]).repeat(32, axis=1), None, out=x)
    np.clip(x, None, np.abs(x[:, ::32]).repeat(32, axis=1), out=x)
    codes, _, values = check_round_trip(ctx, x)
    check.that(
        bool(np.isin(codes[:, ::32] & 7, (6, 7)).all()),
        "the largest magnitude of a block must be encoded as 4 or 6 times the scale",
        f"codes of the block maxima: {codes[:, ::32].tolist()}",
    )
    check.that(
        bool((np.sign(values[:, ::32]) == np.sign(x[:, ::32])).all()),
        "a saturated value must keep its sign",
    )


# Rejects decode_subnormal_as_normal and exponent_wraps_uint8.
@case("mxfp4_decode: all 16 codes, with scales below and above 1", public=True)
def decode_all_codes(ctx):
    codes = np.stack([np.roll(np.arange(16, dtype=np.uint8), k).repeat(4) for k in range(4)])
    exps = np.array([[127, 120], [100, 131], [126, 160], [65, 184]], dtype=np.uint8)
    values = check_decode(ctx, codes, exps)
    zeros = (codes & 7) == 0
    check.that(not values[zeros].any(), "codes 0 and 8 must decode to zero")
    check.that(
        bool((np.abs(values[(codes & 7) == 1]) > 0).all()),
        "codes 1 and 9, the subnormal 0.5, must not decode to zero",
    )


# Rejects drops_sign_of_zero and subnormal_ignored.
@case("all-zero blocks, exact zeros, and small values around the subnormal 0.5", public=True)
def zeros_and_subnormals(ctx):
    x = np.zeros((4, 64), dtype=np.float32)
    x[0, 32:] = ctx.randn(32)
    x[1, :32] = ctx.randn(32) * 1e-3
    small = np.array([0.1, -0.1, 0.24, -0.24, 0.26, -0.3, 0.5, -0.5, 0.6, 0.74, -0.76, 0.9])
    x[2, 0], x[2, 1:13] = 4.0, small
    x[3, 32], x[3, 33:45] = -1024.0, small * 256.0
    x[3, 50:60] = 0.0
    codes, exps, values = check_round_trip(ctx, x)
    check.that(
        int(exps[0, 0]) == 127 and not codes[0, :32].any(),
        "an all-zero block must get scale byte 127 and codes 0",
        f"exps = {int(exps[0, 0])}, codes = {codes[0, :32].tolist()}",
    )
    check.equal(codes[2, 1:5], np.array([0, 8, 0, 8]), label="codes of +-0.1 and +-0.24")
    check.that(not values[x == 0].any(), "a zero must decode to zero")


# Rejects naive_log2.
@case("block maxima that are powers of two, and one ulp below a power of two")
def power_of_two_maxima(ctx):
    x = scaled_blocks(ctx, 4, 2, low=-2.0, high=2.0)
    exponents = [20, -20, 23, 1, 37, -37, 0, 52]
    for i, k in enumerate(exponents):
        row, block = divmod(i, 2)
        top = np.float32(2.0**k)
        if i % 4 < 2:
            top = np.nextafter(top, np.float32(0.0))  # the largest float32 below 2**k
        segment = x[row, 32 * block : 32 * block + 32]
        segment *= top / np.abs(segment).max() / 2
        segment[i % 32] = top if i % 2 else -top
    check_round_trip(ctx, x)


@case("a square (32, 32) matrix, and rows of very different magnitude")
def square_and_extremes(ctx):
    check_round_trip(ctx, scaled_blocks(ctx, 32, 1))
    x = scaled_blocks(ctx, 4, 2, low=-58.0, high=58.0)
    check_round_trip(ctx, x)
    check_round_trip(ctx, np.abs(scaled_blocks(ctx, 4, 2)))
    check_round_trip(ctx, -np.abs(scaled_blocks(ctx, 4, 2)))


@case("mxfp4_decode: random codes and scale bytes over the whole allowed range")
def decode_random(ctx):
    for _ in range(3):
        codes = ctx.randint(0, 16, (4, 64)).astype(np.uint8)
        exps = ctx.randint(60, 191, (4, 2)).astype(np.uint8)
        check_decode(ctx, codes, exps)


@case("decode(encode(x)) is the nearest representable value under the specified scale")
def nearest_property(ctx):
    def prop(rng):
        x = scaled_blocks(ctx, 4, 2, low=float(rng.uniform(-50, 0)), high=float(rng.uniform(0, 50)))
        x[rng.random(x.shape) < 0.1] = 0.0
        if rng.random() < 0.5:
            # Dyadic values: many land exactly on a tie or on a grid point.
            grid = rng.integers(-32, 33, size=x.shape) / 4.0
            x = (grid * 2.0 ** rng.integers(-20, 21, size=(4, 2)).repeat(BLOCK, axis=1)).astype(
                np.float32
            )
        _, exps, values = check_round_trip(ctx, x)
        scale = np.repeat(2.0 ** (exps.astype(np.float64) - 127), BLOCK, axis=1)
        largest = np.abs(x).reshape(4, 2, BLOCK).max(axis=-1)
        ratio = largest / scale[:, ::BLOCK]
        check.that(
            bool(((largest == 0) | ((ratio >= 4) & (ratio < 8))).all()),
            "the block maximum divided by the scale must lie in [4, 8)",
            f"block maximum / scale = {ratio.tolist()}",
        )
        # Every candidate value of the block is at least as far from x as the decoded one,
        # except beyond the largest element value, where the nearest candidate is 6 scales.
        error = np.abs(values.astype(np.float64) - x)
        candidates = MAGNITUDES * scale[..., None]
        best = np.abs(np.abs(x.astype(np.float64))[..., None] - candidates).min(axis=-1)
        check.that(
            bool((error <= best * (1 + 1e-6)).all()),
            "a decoded value is not the representable value nearest to x in its block",
            f"largest excess, in units of the block scale: {((error - best) / scale).max():.4f}",
        )
        check.that(
            bool((error <= 2 * scale).all()),
            "the round-trip error exceeds two scales, the most that saturation can cost",
        )

    check.forall(prop, trials=8)
