"""Tests for dtypes-and-promotion.

Every case names the function it tests. Inputs arrive with the dtype they have here:
int8 and uint8 arrays are passed unchanged, and int64 positions become the framework's
default integer type.

The judge compares integer results exactly and ignores their width, so the uint8
requirement of saturating_add_u8 is checked by `check_u8` below. The float32 result of
dequantize is checked by ctx.check.
"""

import numpy as np

from learnai.judge import case, check

# Accepts q * scale - zero_point * scale as well: over every code, every zero point, and
# 404 scales in [0.001, 1] its largest error was 1.53e-5, in all three frameworks.
DEQ_TOL = {"rtol": 1e-4, "atol": 1e-4}
ALL_CODES = np.arange(-128, 128, dtype=np.int8).reshape(16, 16)


def i8(values):
    return np.array(values, dtype=np.int8)


def u8(values):
    return np.array(values, dtype=np.uint8)


def ints(values):
    return np.array(values, dtype=np.int64)


def floats(values):
    return np.array(values, dtype=np.float32)


def check_u8(ctx, a, b, expected=None):
    """Compare the values with the reference, then require the uint8 dtype."""
    got = ctx.check("saturating_add_u8", a, b)
    check.that(
        got.dtype == np.uint8,
        f"saturating_add_u8: got dtype {got.dtype}, expected dtype uint8",
        "The values are right. The result must be a uint8 tensor: cast back after the clamp.",
    )
    if expected is not None:
        check.equal(got, u8(expected), label="saturating_add_u8")
    return got


def check_blocks(ctx, pos, size, block=None, offset=None):
    got_block, got_offset = ctx.check("block_and_offset", pos, size)
    if block is not None:
        check.equal(got_block, ints(block), label="block_and_offset[0]")
        check.equal(got_offset, ints(offset), label="block_and_offset[1]")


# ------------------------------------------------------------------------ dequantize


# Rejects dequant_subtracts_in_int8 (100 + 100 wraps to -56, giving -28 for 100) and
# dequant_zero_point_unscaled.
@case("dequantize: the worked example, zero_point = -100 and scale = 0.5", public=True)
def dequant_example(ctx):
    got = ctx.check("dequantize", i8([-128, 0, 100, 127]), 0.5, -100, **DEQ_TOL)
    check.allclose(got, floats([-14.0, 50.0, 100.0, 113.5]), label="dequantize", **DEQ_TOL)


# The int8 subtraction is right here: nothing leaves the range.
@case("dequantize: zero_point = 0, q of shape (2, 3)", public=True)
def dequant_zero_point_zero(ctx):
    got = ctx.check("dequantize", i8([[-128, -1, 0], [1, 100, 127]]), 0.25, 0, **DEQ_TOL)
    check.allclose(got, floats([[-32.0, -0.25, 0.0], [0.25, 25.0, 31.75]]), label="dequantize", **DEQ_TOL)


@case("dequantize: every int8 code with zero_point = -128, results 0 to 255", public=True)
def dequant_all_codes_low(ctx):
    got = ctx.check("dequantize", ALL_CODES, 1.0, -128, **DEQ_TOL)
    check.allclose(got, np.arange(256, dtype=np.float32).reshape(16, 16), label="dequantize", **DEQ_TOL)


@case("dequantize: every int8 code with zero_point = 127, results -255 to 0")
def dequant_all_codes_high(ctx):
    got = ctx.check("dequantize", ALL_CODES, 1.0, 127, **DEQ_TOL)
    check.allclose(got, np.arange(-255, 1, dtype=np.float32).reshape(16, 16), label="dequantize", **DEQ_TOL)


@case("dequantize: every int8 code, zero points across the range, scales 0.001 to 1")
def dequant_sweep(ctx):
    for zero_point in (-128, -77, -1, 0, 1, 64, 127):
        scale = float(10.0 ** ctx.rng.uniform(-3.0, 0.0))
        ctx.check("dequantize", ALL_CODES, scale, zero_point, **DEQ_TOL)


@case("dequantize: random codes, shapes (1, 5), (5, 1), (3, 3), and (2, 3, 4)")
def dequant_shapes(ctx):
    for shape in ((1, 5), (5, 1), (3, 3), (2, 3, 4)):
        q = ctx.rng.integers(-128, 128, size=shape).astype(np.int8)
        ctx.check("dequantize", q, 0.05, int(ctx.rng.integers(-128, 128)), **DEQ_TOL)


# ----------------------------------------------------------------- saturating_add_u8


# Rejects add_wraps and add_clamps_after_wrap: 200 + 100 wraps to 44 and 255 + 255
# to 254.
@case("saturating_add_u8: the worked example, sums of 0, 200, 300, and 510", public=True)
def add_example(ctx):
    check_u8(ctx, u8([0, 100, 200, 255]), u8([0, 100, 100, 255]), expected=[0, 200, 255, 255])


# Plain uint8 addition is right here. Rejects add_returns_int16 by its dtype.
@case("saturating_add_u8: no sum above 255, and the result is uint8", public=True)
def add_in_range(ctx):
    a = u8([[1, 2, 3], [100, 127, 0]])
    b = u8([[10, 20, 30], [100, 128, 255]])
    check_u8(ctx, a, b, expected=[[11, 22, 33], [200, 255, 255]])


@case("saturating_add_u8: sums of exactly 255 and 256, shape (1, 5)", public=True)
def add_boundary(ctx):
    a = u8([[127, 128, 255, 255, 1]])
    b = u8([[128, 128, 0, 1, 255]])
    check_u8(ctx, a, b, expected=[[255, 255, 255, 255, 255]])


@case("saturating_add_u8: a (16, 16) grid of every pair of multiples of 17")
def add_grid(ctx):
    steps = np.arange(0, 256, 17, dtype=np.uint8)
    a = np.repeat(steps[:, None], 16, axis=1)
    b = np.repeat(steps[None, :], 16, axis=0)
    got = check_u8(ctx, a, b)
    check.equal(got[15], u8([255] * 16), label="saturating_add_u8[15]")  # 255 + anything


@case("saturating_add_u8: random values, shapes (1, 5), (5, 1), (3, 3), and (2, 3, 4)")
def add_shapes(ctx):
    for shape in ((1, 5), (5, 1), (3, 3), (2, 3, 4)):
        a = ctx.rng.integers(0, 256, size=shape).astype(np.uint8)
        b = ctx.rng.integers(0, 256, size=shape).astype(np.uint8)
        check_u8(ctx, a, b)


# ------------------------------------------------------------------ block_and_offset


# Rejects block_truncates (-7 // 3 gives -2 for -3) and offset_sign_of_dividend
# (-7 gives -1 for 2).
@case("block_and_offset: the worked example, positions -7 to 7 and size = 3", public=True)
def blocks_example(ctx):
    check_blocks(
        ctx,
        ints([-7, -6, -1, 0, 1, 6, 7]),
        3,
        block=[-3, -2, -1, 0, 0, 2, 2],
        offset=[2, 0, 2, 0, 1, 0, 1],
    )


# Truncation and floor agree here.
@case("block_and_offset: positions 0 to 9 and size = 4, shape (2, 5)", public=True)
def blocks_non_negative(ctx):
    check_blocks(
        ctx,
        np.arange(10, dtype=np.int64).reshape(2, 5),
        4,
        block=[[0, 0, 0, 0, 1], [1, 1, 1, 2, 2]],
        offset=[[0, 1, 2, 3, 0], [1, 2, 3, 0, 1]],
    )


@case("block_and_offset: negative positions with size = 4, size = 1, and size = 100", public=True)
def blocks_negative(ctx):
    pos = ints([-8, -5, -4, -3, -1])
    check_blocks(ctx, pos, 4, block=[-2, -2, -1, -1, -1], offset=[0, 3, 0, 1, 3])
    check_blocks(ctx, pos, 1, block=[-8, -5, -4, -3, -1], offset=[0, 0, 0, 0, 0])
    check_blocks(ctx, pos, 100, block=[-1, -1, -1, -1, -1], offset=[92, 95, 96, 97, 99])


# Rejects block_by_float_division: float32 holds 999999999 as 1000000000.
@case("block_and_offset: positions up to 1e9 in magnitude, sizes 1, 7, 1000, and 1000000")
def blocks_large(ctx):
    pos = ints(
        [999_999_999, -999_999_999, 1_000_000_000, -1_000_000_000, 16_777_217, -16_777_217, 123_456_789]
    )
    check_blocks(
        ctx,
        pos,
        1000,
        block=[999_999, -1_000_000, 1_000_000, -1_000_000, 16_777, -16_778, 123_456],
        offset=[999, 1, 0, 0, 217, 783, 789],
    )
    for size in (1, 7, 1_000_000):
        check_blocks(ctx, pos, size)


@case("block_and_offset: random positions in [-1000, 1000], four shapes, five sizes")
def blocks_shapes(ctx):
    for shape in ((1, 5), (5, 1), (3, 3), (2, 3, 4)):
        pos = ctx.randint(-1000, 1001, shape)
        for size in (1, 2, 5, 16, 2048):
            check_blocks(ctx, pos, size)


@case("block_and_offset: exact multiples of size on both sides of zero, size = 16")
def blocks_multiples(ctx):
    pos = 16 * np.arange(-3, 4, dtype=np.int64)
    check_blocks(ctx, pos, 16, block=[-3, -2, -1, 0, 1, 2, 3], offset=[0] * 7)
