"""Tests for quant-error-metrics. Every case compares the solution with the oracle.

`quantized(ctx, x, bits)` makes a realistic approximation: x rounded onto a uniform grid
of 2**bits levels spanning its range, so the error has the size and the structure of
real quantization noise.
"""

import numpy as np

from whetstone.judge import case, check

ELEMENTWISE = ("mse", "sqnr_db", "max_abs_error")


def quantized(x, bits):
    """x rounded to a uniform grid of 2**bits levels over [min(x), max(x)]."""
    lo, hi = float(x.min()), float(x.max())
    step = (hi - lo) / (2**bits - 1)
    return (np.round((x - lo) / step) * step + lo).astype(np.float32)


def check_scalar(ctx, name, *args, **tolerance):
    """The metric must be a 0-d float32 tensor equal to the oracle's."""
    got = ctx.check(name, *args, **tolerance)
    check.shape(got, (), label=name)
    return float(got)


def check_elementwise(ctx, a, b):
    for name in ELEMENTWISE:
        check_scalar(ctx, name, a, b)


# output_error subtracts two matrix products that agree to a few percent, so it is
# compared to 1e-3 (the statement says so); the other metrics use the default 1e-4.
def check_output_error(ctx, w, w_hat, x):
    return check_scalar(ctx, "output_error", w, w_hat, x, rtol=1e-3)


@case("the worked example: four numbers, and a 2 x 3 layer on four inputs", public=True)
def worked_example(ctx):
    a = np.array([1.0, -2.0, 3.0, 0.5], dtype=np.float32)
    b = np.array([1.1, -2.0, 2.8, 0.5], dtype=np.float32)
    check.allclose(ctx.call("mse", a, b), np.float32(0.0125), label="mse")
    check.allclose(ctx.call("sqnr_db", a, b), np.float32(24.548447), label="sqnr_db")
    check.allclose(ctx.call("max_abs_error", a, b), np.float32(0.2), label="max_abs_error")
    w = np.array([[1.0, 2.0, -1.0], [0.5, -0.5, 2.0]], dtype=np.float32)
    w_hat = np.array([[1.0, 2.25, -1.0], [0.5, -0.5, 1.75]], dtype=np.float32)
    x = np.array(
        [[1.0, 0.0, 1.0], [2.0, 1.0, 0.0], [0.0, -1.0, 1.0], [1.0, 1.0, 1.0]], dtype=np.float32
    )
    check.allclose(
        ctx.call("output_error", w, w_hat, x), np.float32(0.0905357), label="output_error"
    )
    check_elementwise(ctx, a, b)
    check_elementwise(ctx, w, w_hat)
    check_output_error(ctx, w, w_hat, x)


# Rejects mse_last_axis: a 2-D input must still give one number.
@case("a (16, 32) weight matrix and its 4-bit quantization", public=True)
def quantized_matrix(ctx):
    w = ctx.randn(16, 32, scale=0.05)
    w_hat = quantized(w, 4)
    check_elementwise(ctx, w, w_hat)
    check_output_error(ctx, w, w_hat, ctx.randn(24, 32))


# Rejects sqnr_20log: 10 log10(100) is 20.
@case("an approximation that is off by 10% everywhere has an SQNR of 20 dB", public=True)
def twenty_decibels(ctx):
    ref = ctx.randn(8, 8, scale=2.0)
    approx = (ref * np.float32(1.1)).astype(np.float32)
    got = check_scalar(ctx, "sqnr_db", ref, approx)
    check.that(abs(got - 20.0) < 0.01, "a 10% error must give 20 dB", f"got {got:.4f} dB")
    got = check_scalar(ctx, "sqnr_db", ref, (ref * np.float32(0.999)).astype(np.float32))
    check.that(abs(got - 60.0) < 0.05, "a 0.1% error must give 60 dB", f"got {got:.4f} dB")


# Rejects sqnr_epsilon.
@case("an exact approximation: zero error, and an SQNR of +inf", public=True)
def exact_approximation(ctx):
    ref = ctx.randn(5, 7)
    check_elementwise(ctx, ref, ref.copy())
    got = ctx.call("sqnr_db", ref, ref.copy())
    check.that(
        bool(np.isposinf(got)),
        "sqnr_db must be +inf when approx equals ref",
        f"got {got}",
    )
    check.that(float(ctx.call("mse", ref, ref.copy())) == 0.0, "mse(a, a) must be exactly 0")
    check_output_error(ctx, ref, quantized(ref, 3), ctx.randn(6, 7))


# Rejects max_error_signed: here a - b is never positive.
@case("the largest error is negative: b is above a everywhere")
def one_sided_error(ctx):
    a = ctx.randn(6, 10)
    b = (a + ctx.uniform(6, 10, low=0.01, high=0.5)).astype(np.float32)
    check_elementwise(ctx, a, b)
    check_elementwise(ctx, b, a)
    a = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    check_elementwise(ctx, a, np.array([0.1, 1.0, 5.0, 2.9], dtype=np.float32))


# Rejects no_transpose (which only runs here) and normalized_by_approx.
@case("output_error with a square weight and a coarse, 2-bit approximation")
def square_and_coarse(ctx):
    w = ctx.randn(12, 12, scale=0.3)
    x = ctx.randn(12, 12)
    for bits in (2, 3, 6):
        check_output_error(ctx, w, quantized(w, bits), x)
    # An approximation that is uniformly too small: the two outputs have different norms.
    check_output_error(ctx, w, (w * np.float32(0.6)).astype(np.float32), x)


# Rejects output_error_per_row: a few rows of x dominate both norms.
@case("output_error when the rows of x have very different magnitudes")
def uneven_inputs(ctx):
    w = ctx.randn(8, 20, scale=0.1)
    x = ctx.randn(10, 20) * np.array([[100.0], [30.0], *[[0.05]] * 8], dtype=np.float32)
    check_output_error(ctx, w, quantized(w, 3), x)
    check_output_error(ctx, w, quantized(w, 5), x[::-1].copy())


@case("output_error with one input row, one output row, and more outputs than inputs")
def layer_shapes(ctx):
    w = ctx.randn(5, 9)
    check_output_error(ctx, w, quantized(w, 4), ctx.randn(1, 9))
    w = ctx.randn(1, 9)
    check_output_error(ctx, w, quantized(w, 3), ctx.randn(7, 9))
    w = ctx.randn(11, 4)
    check_output_error(ctx, w, quantized(w, 4), ctx.randn(3, 4))


@case("1-D, 3-D, and 4-D tensors reduce to a single number")
def any_shape(ctx):
    for shape in ((37,), (3, 4, 5), (2, 3, 4, 5)):
        ref = ctx.randn(*shape, scale=3.0)
        check_elementwise(ctx, ref, quantized(ref, 5))


@case("signals and errors on very different scales")
def many_scales(ctx):
    for scale, bits in ((1e-3, 8), (1.0, 2), (250.0, 6), (0.02, 3)):
        ref = ctx.randn(8, 16, scale=scale, loc=scale)
        check_elementwise(ctx, ref, quantized(ref, bits))
