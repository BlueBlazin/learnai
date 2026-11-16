"""Tests for nf4-codebook.

Codes are compared exactly, so no normalized weight may sit near the midpoint between
two adjacent levels: a level table that differs in the last float32 bit could then pick
the other neighbor. Every weight matrix goes through `midpoint_free`, which keeps
w / absmax at least 0.001 away from every midpoint, as the statement promises. The
published table is the ground truth here: nearest codes are recomputed from it in
numpy, independently of any reference.

JAX compiles every new combination of shape and block size, so the cases share a few
layouts.
"""

import numpy as np

from learnai.judge import case, check

# The table hard-coded in bitsandbytes (functional.py), reproduced by its
# create_normal_map(offset=0.9677083) in float64.
LEVELS = np.array(
    [
        -1.0,
        -0.6961928009986877,
        -0.5250730514526367,
        -0.39491748809814453,
        -0.28444138169288635,
        -0.18477343022823334,
        -0.09105003625154495,
        0.0,
        0.07958029955625534,
        0.16093020141124725,
        0.24611230194568634,
        0.33791524171829224,
        0.44070982933044434,
        0.5626170039176941,
        0.7229568362236023,
        1.0,
    ]
)
MIDPOINTS = (LEVELS[1:] + LEVELS[:-1]) / 2
MARGIN = 1e-3
# Half the widest gap between adjacent levels (codes 0 and 1): the worst rounding error.
WORST_HALF_GAP = float(np.diff(LEVELS).max()) / 2


def block_absmax(w, block_size):
    """(blocks, absmax) of the statement, in float32 like a solution computes them."""
    out_dim, in_dim = w.shape
    blocks = w.reshape(out_dim, in_dim // block_size, block_size)
    absmax = np.abs(blocks).max(axis=-1, keepdims=True)
    return blocks, np.where(absmax > 0, absmax, np.float32(1.0)).astype(np.float32)


def normalized(w, block_size):
    blocks, absmax = block_absmax(w, block_size)
    return blocks.astype(np.float64) / absmax


def nearest_codes(w, block_size):
    """The index of the nearest published level for every weight."""
    n = normalized(w, block_size)
    return np.abs(n[..., None] - LEVELS).argmin(axis=-1).reshape(w.shape)


def midpoint_free(w, block_size):
    """A copy of w in which no normalized weight is within MARGIN of a midpoint: the
    offenders are moved onto the level just below. The absmax of a block is never moved
    (it normalizes to +-1, which is a level)."""
    w = np.array(w, dtype=np.float32)
    blocks, absmax = block_absmax(w, block_size)
    n = blocks.astype(np.float64) / absmax
    gap = np.abs(n[..., None] - MIDPOINTS)
    bad = gap.min(axis=-1) < MARGIN
    lower = LEVELS[gap.argmin(axis=-1)]
    fixed = np.where(bad, (lower * absmax).astype(np.float32), blocks).reshape(w.shape)
    n = normalized(fixed, block_size)
    if (np.abs(n[..., None] - MIDPOINTS).min(axis=-1) < MARGIN).any():
        raise AssertionError("could not move the input away from the midpoints")
    assert np.array_equal(block_absmax(fixed, block_size)[1], absmax)
    return fixed


def dequantize(codes, absmax, block_size):
    """levels[codes] * absmax in float64, from the published table."""
    return LEVELS[codes.astype(np.int64)] * np.repeat(absmax.astype(np.float64), block_size, axis=1)


def check_quantize(ctx, w, block_size):
    """nf4_quantize against the oracle and against the published table."""
    w = midpoint_free(w, block_size)
    codes, absmax = ctx.check("nf4_quantize", w, block_size)
    check.that(codes.dtype == np.uint8, "codes must be a uint8 tensor", f"dtype {codes.dtype}")
    check.equal(codes, nearest_codes(w, block_size), label="codes")
    return w, codes, absmax


def native_round_trip(module, fw, w, block_size):
    """nf4_quantize then nf4_dequantize on framework tensors, so the codes and scales
    reach the dequantizer exactly as the quantizer returned them."""
    codes, absmax = module.nf4_quantize(fw.tensor(w), block_size)
    values = module.nf4_dequantize(codes, absmax, block_size)
    return fw.numpy(codes), fw.numpy(absmax), fw.numpy(values)


def blocks_of(ctx, out_dim, n_blocks, block_size, *, low=-3.0, high=1.0):
    """A matrix whose blocks have magnitudes of their own, from 10**low to 10**high."""
    magnitude = 10.0 ** ctx.uniform(out_dim, n_blocks, 1, low=low, high=high)
    w = ctx.randn(out_dim, n_blocks, block_size) * magnitude
    return w.reshape(out_dim, n_blocks * block_size).astype(np.float32)


WORKED = np.array(
    [
        [0.0, 0.1, -0.25, 0.52, -1.0, 0.33, 0.75, -0.6],
        [0.01, -0.04, 0.0, 0.018, -0.013, 0.033, -0.003, 0.004],
    ],
    dtype=np.float32,
)


@case("the worked example: two rows of one block each", public=True)
def worked_example(ctx):
    codes, absmax = ctx.call("nf4_quantize", WORKED, 8)
    check.allclose(absmax, np.array([[1.0], [0.04]], np.float32), label="absmax")
    check.equal(
        codes,
        np.array([[7, 8, 4, 13, 0, 11, 14, 2], [10, 0, 7, 12, 4, 14, 6, 8]]),
        label="codes",
    )
    check_quantize(ctx, WORKED, 8)
    want = np.array(
        [
            [0.0, 0.079580, -0.284441, 0.562617, -1.0, 0.337915, 0.722957, -0.525073],
            [0.009844, -0.04, 0.0, 0.017628, -0.011378, 0.028918, -0.003642, 0.003183],
        ],
        dtype=np.float32,
    )
    values = ctx.check("nf4_dequantize", codes.astype(np.uint8), absmax, 8)
    check.allclose(values, want, rtol=1e-3, label="nf4_dequantize")


# Rejects uniform_levels, symmetric_levels, and levels_not_normalized.
@case("nf4_levels: the published table, ascending, with an exact zero", public=True)
def levels_table(ctx):
    levels = ctx.call("nf4_levels")
    check.shape(levels, (16,), label="nf4_levels")
    check.allclose(levels, LEVELS.astype(np.float32), dtype=True, label="nf4_levels")
    check.that(
        float(levels[7]) == 0.0,
        "level 7 must be exactly 0.0",
        f"level 7 = {float(levels[7])!r}",
    )
    check.that(bool((np.diff(levels) > 0).all()), "the levels must be in ascending order")


# Rejects nearest_by_index, std_normalization, and global_absmax.
@case("a (4, 64) matrix in blocks of 16, each block on its own scale", public=True)
def blocks_with_own_scales(ctx):
    check_quantize(ctx, blocks_of(ctx, 4, 4, 16), 16)


# Rejects zero_block_nan, and tables without an exact zero.
@case("zero weights and all-zero blocks stay exactly zero", public=True)
def zeros(ctx):
    w = blocks_of(ctx, 4, 4, 16)
    w[ctx.rng.random(w.shape) < 0.3] = 0.0
    w[1, 16:32] = 0.0
    w[3, 0:16] = 0.0
    w, codes, absmax = check_quantize(ctx, w, 16)
    check.finite(absmax, label="absmax")
    check.that(
        float(absmax[1, 1]) == 1.0 and float(absmax[3, 0]) == 1.0,
        "an all-zero block must get absmax 1",
        f"absmax = {absmax}",
    )
    check.that(bool((codes[w == 0] == 7).all()), "a weight of 0 must get code 7, the zero level")
    _, _, values = native_round_trip(ctx.solution, ctx.fw, w, 16)
    check.that(
        not values[w == 0].any(),
        "a weight of 0 must come back as exactly 0.0",
        f"largest magnitude returned for a zero weight: {np.abs(values[w == 0]).max()!r}",
    )
    check_quantize(ctx, np.zeros((4, 64), dtype=np.float32), 16)


# Rejects absmax_tiled.
@case("nf4_dequantize: every code, with a different scale for every block", public=True)
def dequantize_given_codes(ctx):
    codes = np.stack([np.roll(np.arange(16, dtype=np.uint8), k).repeat(4) for k in range(4)])
    absmax = (10.0 ** ctx.uniform(4, 4, low=-3, high=1)).astype(np.float32)
    values = ctx.check("nf4_dequantize", codes, absmax, 16)
    check.allclose(values, dequantize(codes, absmax, 16), label="nf4_dequantize")
    check.that(not values[codes == 7].any(), "code 7 must dequantize to exactly 0.0")


@case("the default layout: blocks of 64, small weights")
def blocks_of_64(ctx):
    check_quantize(ctx, ctx.randn(4, 64, scale=0.02), 64)
    check_quantize(ctx, blocks_of(ctx, 4, 1, 64), 64)


@case("one outlier only costs its own block")
def outlier(ctx):
    w = ctx.randn(4, 64, scale=0.05)
    w[1, 40] = 9.0
    w[3, 63] = -30.0
    check_quantize(ctx, w, 16)


@case("blocks of one sign: the extreme is the most negative or the most positive weight")
def one_sign(ctx):
    w = np.abs(blocks_of(ctx, 4, 4, 16)) + np.float32(1e-3)
    _, codes, _ = check_quantize(ctx, w, 16)
    check.that(
        int(codes.min()) >= 7 and int(codes.max()) == 15,
        "the codes of a block lie on the wrong side of the zero level",
    )
    _, codes, _ = check_quantize(ctx, -w, 16)
    check.that(
        int(codes.max()) <= 7 and int(codes.min()) == 0,
        "the codes of a block lie on the wrong side of the zero level",
    )


@case("a square matrix, and block_size 1")
def other_layouts(ctx):
    check_quantize(ctx, blocks_of(ctx, 16, 1, 16), 16)
    # Blocks of one element: every non-zero weight is its own absmax and gets code 0 or 15.
    w = ctx.randn(2, 8)
    w[0, 3] = 0.0
    _, codes, absmax = check_quantize(ctx, w, 1)
    check.that(
        bool(np.isin(codes, (0, 7, 15)).all()),
        "the absmax of a block must get code 0 or 15, and a zero weight code 7",
    )
    check.allclose(absmax, np.where(w == 0, 1.0, np.abs(w)).astype(np.float32), label="absmax")


@case("every weight comes back as the nearest level of its block, within the widest half gap")
def nearest_property(ctx):
    layouts = [(4, 4, 16), (4, 1, 64), (16, 1, 16)]

    def prop(rng):
        out_dim, n_blocks, block_size = layouts[int(rng.integers(len(layouts)))]
        low = float(rng.uniform(-4, 0))
        w = midpoint_free(blocks_of(ctx, out_dim, n_blocks, block_size, low=low), block_size)
        if rng.random() < 0.5:
            w = (w**3 / np.abs(w).max() ** 2).astype(np.float32)  # heavy tails
            w = midpoint_free(w, block_size)
        codes, absmax, values = native_round_trip(ctx.solution, ctx.fw, w, block_size)
        check.shape(codes, w.shape, label="codes")
        check.shape(absmax, (out_dim, n_blocks), label="absmax")
        check.shape(values, w.shape, label="nf4_dequantize(*nf4_quantize(w))")
        check.that(
            codes.dtype == np.uint8 and int(codes.max()) <= 15,
            "codes must be uint8 values from 0 to 15",
            f"dtype {codes.dtype}, largest code {int(codes.max())}",
        )
        check.allclose(absmax, block_absmax(w, block_size)[1][..., 0], label="absmax")
        check.equal(codes, nearest_codes(w, block_size), label="codes")
        check.allclose(values, dequantize(codes, absmax, block_size), label="round trip")
        scale = np.repeat(absmax.astype(np.float64), block_size, axis=1)
        error = np.abs(values.astype(np.float64) - w)
        check.that(
            bool((error <= WORST_HALF_GAP * scale * 1.001 + 1e-7).all()),
            "a weight is further from its dequantized value than half the widest gap of the table",
            f"largest error in units of absmax: {(error / scale).max():.4f}",
        )

    check.forall(prop, trials=8)
