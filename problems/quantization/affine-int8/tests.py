"""Tests for affine-int8.

Codes are compared exactly, so no test input may sit near a rounding tie: frameworks
could then round x / scale to different sides because of a one-ulp difference. Every
input goes through `tie_free`, which keeps both x / scale and the zero-point at least
0.05 away from a half-integer.
"""

import numpy as np

from whetstone.judge import case, check

QMIN, QMAX = -128, 127
MARGIN = 0.05


def near_tie(v):
    """True where v (float64) is within MARGIN of a half-integer."""
    return np.abs(v - np.floor(v) - 0.5) < MARGIN


def grid(x):
    """(lo, hi, scale) of the statement, computed in float32 like a solution would."""
    lo, hi = np.float32(min(x.min(), 0.0)), np.float32(max(x.max(), 0.0))
    scale = (hi - lo) / np.float32(QMAX - QMIN) if hi > lo else np.float32(1.0)
    return lo, hi, np.float32(scale)


def tie_free(ctx, make):
    """Draw x = make() until its zero-point is clear of a tie, then move every element
    whose code would be near a tie onto the nearest grid point."""
    for _ in range(200):
        x = np.asarray(make(), dtype=np.float32)
        lo, hi, scale = grid(x)
        if near_tie(np.float64(lo) / np.float64(scale)):
            continue
        t = x.astype(np.float64) / np.float64(scale)
        bad = near_tie(t)
        x[bad] = np.clip(np.round(t[bad]) * np.float64(scale), lo, hi).astype(np.float32)
        t = x.astype(np.float64) / np.float64(scale)
        if grid(x) == (lo, hi, scale) and not near_tie(t).any():
            return x
    raise AssertionError("could not draw a tie-free input")


def native_round_trip(module, fw, x, probe=None):
    """quant_params(x) -> quantize -> dequantize on framework tensors, so the int8 codes
    reach dequantize with their own dtype. Quantizes `probe` (default: x itself) on the
    grid of x. Returns numpy (codes, values)."""
    xt = fw.tensor(x)
    scale, zero_point = module.quant_params(xt)
    q = module.quantize(xt if probe is None else fw.tensor(probe), scale, zero_point)
    return fw.numpy(q), fw.numpy(module.dequantize(q, scale, zero_point))


def check_all(ctx, x):
    """Check the three functions separately, then chained on the solution's own outputs."""
    ctx.check("quant_params", x)
    scale, zero_point = ctx.expected("quant_params", x)
    ctx.check("quantize", x, scale, zero_point)
    q = ctx.expected("quantize", x, scale, zero_point)
    ctx.check("dequantize", q, scale, zero_point)

    codes, values = native_round_trip(ctx.solution, ctx.fw, x)
    want_codes, want_values = native_round_trip(ctx.oracle, ctx.oracle_fw, x)
    check.that(
        codes.dtype == np.int8,
        "quantize must return an int8 tensor",
        f"got dtype {codes.dtype}",
    )
    check.equal(codes, want_codes, label="quantize(x, *quant_params(x))")
    check.allclose(values, want_values, label="dequantize(quantize(x)), the round trip")

    # Whatever the range of x, 0.0 is a grid point: it must come back as exactly 0.0.
    _, zero = native_round_trip(ctx.solution, ctx.fw, x, probe=np.zeros(1, dtype=np.float32))
    check.that(
        zero.shape == (1,) and float(zero[0]) == 0.0,
        "0.0 is not exactly representable on the grid returned by quant_params",
        f"x = {x}\ndequantize(quantize(0.0)) = {zero}",
    )
    return float(scale), int(zero_point), values


@case("the worked example: x = [-0.5, 0.0, 0.3, 1.2, 2.0]", public=True)
def worked_example(ctx):
    x = np.array([-0.5, 0.0, 0.3, 1.2, 2.0], dtype=np.float32)
    scale, zero_point = ctx.call("quant_params", x)
    check.allclose(scale, np.float32(2.5 / 255), label="scale")
    check.equal(int(zero_point), -77, label="zero_point")
    q = ctx.call("quantize", x, np.array(2.5 / 255, dtype=np.float32), np.array(-77))
    check.equal(q, np.array([-128, -77, -46, 45, 127]), label="quantize")
    check_all(ctx, x)


@case("a random tensor with both signs, shape (4, 8)", public=True)
def mixed_signs(ctx):
    check_all(ctx, tie_free(ctx, lambda: ctx.randn(4, 8, scale=2.0)))


# Rejects range_excludes_zero.
@case("an all-positive tensor in [2, 5]: the range is widened down to 0", public=True)
def all_positive(ctx):
    x = tie_free(ctx, lambda: ctx.uniform(3, 6, low=2.0, high=5.0))
    _, zero_point, _ = check_all(ctx, x)
    check.equal(zero_point, QMIN, label="zero_point of an all-positive tensor")


# Rejects no_clamp and truncates: fixed parameters, values on both sides of the range.
@case("values beyond the grid saturate at -128 and 127", public=True)
def saturation(ctx):
    scale = np.array(0.05, dtype=np.float32)
    codes = np.array([-300, -200, -129, -128, -127, -1, 0, 1, 126, 127, 128, 200, 300, 4000])
    offsets = ctx.uniform(codes.size, low=-0.35, high=0.35)
    for zero_point in (0, -128, 127, 17):
        x = ((codes - zero_point + offsets) * 0.05).astype(np.float32)
        ctx.check("quantize", x, scale, np.array(zero_point))


@case("0.0 survives the round trip exactly, inside and outside the range of x", public=True)
def zero_is_exact(ctx):
    for low, high in ((-1.0, 3.0), (0.37, 0.91), (-7.3, -0.2), (-1e-3, 5e3)):

        def make(low=low, high=high):
            x = ctx.uniform(12, low=low, high=high)
            x[5] = 0.0
            return x

        x = tie_free(ctx, make)
        _, values = native_round_trip(ctx.solution, ctx.fw, x)
        check.shape(values, x.shape, label="dequantize")
        check.that(
            float(values[5]) == 0.0,
            "0.0 did not come back as exactly 0.0 after quantize and dequantize",
            f"x = {x}\nround trip = {values}",
        )
        check_all(ctx, x)
    # 0.0 is not an element here, yet it must still be a grid point.
    check_all(ctx, tie_free(ctx, lambda: ctx.uniform(12, low=1.5, high=1.9)))
    check_all(ctx, tie_free(ctx, lambda: ctx.uniform(12, low=-30.0, high=-20.0)))


@case("an all-negative tensor: the range is widened up to 0")
def all_negative(ctx):
    x = tie_free(ctx, lambda: ctx.uniform(2, 3, 4, low=-9.0, high=-4.0))
    _, zero_point, _ = check_all(ctx, x)
    check.equal(zero_point, QMAX, label="zero_point of an all-negative tensor")


@case("constant tensors: all zeros, one positive value, one negative value")
def constant_tensors(ctx):
    zeros = np.zeros((2, 3), dtype=np.float32)
    scale, _, values = check_all(ctx, zeros)
    check.that(scale == 1.0, "an all-zero tensor must get scale 1", f"got scale {scale}")
    check.that(not values.any(), "an all-zero tensor must round-trip to zeros")
    check_all(ctx, np.full((5,), 3.7, dtype=np.float32))
    check_all(ctx, np.full((2, 2), -0.0625, dtype=np.float32))
    check_all(ctx, np.array([42.0], dtype=np.float32))


# Rejects int8_overflow: q - zero_point reaches +-255.
@case("dequantize every int8 code, with zero-points -128, 127, and 5")
def every_code(ctx):
    codes = np.arange(QMIN, QMAX + 1, dtype=np.int8).reshape(16, 16)
    scale = np.array(0.02, dtype=np.float32)
    for zero_point in (-128, 127, 5):
        want = ctx.oracle_fw.numpy(
            ctx.oracle.dequantize(
                ctx.oracle_fw.tensor(codes, dtype=np.int8),
                ctx.oracle_fw.tensor(scale),
                ctx.oracle_fw.tensor(np.array(zero_point)),
            )
        )
        got = ctx.fw.numpy(
            ctx.solution.dequantize(
                ctx.fw.tensor(codes, dtype=np.int8),
                ctx.fw.tensor(scale),
                ctx.fw.tensor(np.array(zero_point)),
            )
        )
        check.allclose(got, want, dtype=True, label=f"dequantize with zero_point {zero_point}")
        ctx.check("dequantize", codes.astype(np.int64), scale, np.array(zero_point))


# Rejects zero_point_sign, zero_point_not_rounded, and divides_by_256 on many ranges.
@case("ranges of very different sizes and offsets")
def many_ranges(ctx):
    for low, high in ((-1e-3, 2e-3), (-250.0, 3.0), (-0.3, 4000.0), (-1.0, 1.0), (-6.0, 0.5)):
        check_all(
            ctx, tie_free(ctx, lambda low=low, high=high: ctx.uniform(24, low=low, high=high))
        )


@case("one grid for the whole tensor: a 4-D tensor whose slices have different ranges")
def per_tensor(ctx):
    def make():
        return ctx.randn(2, 3, 4, 5) * ctx.uniform(2, 3, 1, 1, low=0.1, high=8.0)

    check_all(ctx, tie_free(ctx, make))
    check_all(ctx, tie_free(ctx, lambda: ctx.randn(6, 6, scale=0.3, loc=0.2)))


@case("in-range values come back within half a step")
def half_step_error(ctx):
    def prop(rng):
        spread = float(10.0 ** rng.uniform(-2, 3))
        center = float(rng.uniform(-1.5, 1.5)) * spread
        x = tie_free(ctx, lambda: ctx.randn(4, 16, scale=spread, loc=center))
        scale, _ = ctx.call("quant_params", x)
        check.that(float(scale) > 0, "scale must be positive", f"got scale {scale}")
        _, values = native_round_trip(ctx.solution, ctx.fw, x)
        check.shape(values, x.shape, label="dequantize")
        error = float(np.max(np.abs(values.astype(np.float64) - x)))
        check.that(
            error <= 0.5 * float(scale) * 1.001,
            "the round-trip error exceeds half a quantization step",
            f"max |dequantize(quantize(x)) - x| = {error:.6g}, scale = {float(scale):.6g}",
        )

    check.forall(prop, trials=6)
