"""Tests for layernorm. Numeric cases compare with the oracle; two property cases state
what the function means without it.

Inputs keep |mean| within about ten standard deviations (or are exactly constant), so
that float32 rounding of the mean stays far below the tolerance in every framework.
"""

import numpy as np

from whetstone.judge import case, check

ATOL = 1e-5


def params(ctx, d):
    """A non-trivial gain (both signs, order 1) and bias."""
    sign = np.where(ctx.rng.random(d) < 0.25, -1, 1).astype(np.float32)
    return ctx.uniform(d, low=0.5, high=1.5) * sign, ctx.uniform(d, low=-1.0, high=1.0)


def plain(d):
    """Unit gain and zero bias."""
    return np.ones(d, dtype=np.float32), np.zeros(d, dtype=np.float32)


def run(ctx, x, weight, bias, *args, **kwargs):
    return ctx.check("layer_norm", x, weight, bias, *args, atol=ATOL, **kwargs)


@case("the worked example: x = [1, 2, 3, 6] with a gain and a bias", public=True)
def worked_example(ctx):
    weight = np.array([1.0, 1.0, 2.0, 0.5], dtype=np.float32)
    bias = np.array([0.0, 1.0, 0.0, -1.0], dtype=np.float32)
    x = np.array([1.0, 2.0, 3.0, 6.0], dtype=np.float32)
    want = np.array([-1.069043, 0.465478, 0.0, -0.198217], dtype=np.float32)
    got = run(ctx, x, weight, bias)
    check.allclose(got, want, atol=ATOL, label="layer_norm")
    shifted = run(ctx, x + np.float32(100.0), weight, bias)
    check.allclose(shifted, want, atol=ATOL, label="layer_norm of the shifted vector")
    constant = run(ctx, np.full(4, 5.0, dtype=np.float32), weight, bias)
    check.allclose(constant, bias, atol=ATOL, label="layer_norm of the constant vector")


# Rejects unbiased_variance most visibly: at d = 4 the factor is sqrt(3 / 4).
@case("short vectors: shape (6, 4), where dividing by d - 1 is far off", public=True)
def short_vectors(ctx):
    run(ctx, ctx.randn(6, 4, scale=2.0), *plain(4))
    run(ctx, ctx.randn(6, 4, scale=2.0), *params(ctx, 4))


# Rejects normalizes_over_sequence: the tokens have very different means and scales.
@case("batch and sequence axes: shape (2, 5, 16), each token normalized on its own", public=True)
def batch_and_sequence(ctx):
    x = ctx.randn(2, 5, 16) * ctx.uniform(2, 5, 1, low=0.2, high=8.0)
    x = x + ctx.uniform(2, 5, 1, low=-2.0, high=2.0)
    run(ctx, x, *params(ctx, 16))


# Rejects no_mean_subtraction and uncentered_numerator.
@case("activations with a large mean: the mean is removed, the bias is not", public=True)
def large_mean(ctx):
    run(ctx, ctx.randn(4, 8, loc=6.0), *params(ctx, 8))
    run(ctx, ctx.randn(2, 5, 16, loc=-9.0, scale=2.0), *plain(16))


# Rejects eps_outside_sqrt and no_eps: here the variance is comparable to eps.
@case("small activations: rows of size 1e-3 to 3e-2, where eps matters", public=True)
def small_activations(ctx):
    scales = np.array([[1e-3], [3e-3], [1e-2], [3e-2]], dtype=np.float32)
    x = ctx.randn(4, 8) * scales
    run(ctx, x, *plain(8))
    run(ctx, x, *params(ctx, 8))


# Rejects no_eps (0 / 0) and bias_before_gain (the bias must come out unscaled).
@case("constant vectors and d = 1 come out as the bias")
def constant_vectors(ctx):
    x = np.repeat(np.array([[2.5], [-7.0], [0.0], [100.0]], dtype=np.float32), 8, axis=1)
    weight, bias = params(ctx, 8)
    got = run(ctx, x, weight, bias)
    check.finite(got, label="layer_norm")
    check.allclose(got, np.broadcast_to(bias, x.shape), atol=ATOL, label="layer_norm")
    one = run(
        ctx, ctx.randn(5, 1, scale=3.0), np.array([1.5], np.float32), np.array([-0.25], np.float32)
    )
    check.allclose(one, np.full((5, 1), -0.25, np.float32), atol=ATOL, label="layer_norm, d = 1")


@case("the gain is applied before the bias is added")
def gain_then_bias(ctx):
    x = ctx.randn(4, 8, scale=2.0)
    weight = np.linspace(-2.0, 3.0, 8, dtype=np.float32)
    bias = np.linspace(4.0, -4.0, 8, dtype=np.float32)
    run(ctx, x, weight, bias)
    run(ctx, x, np.zeros(8, dtype=np.float32), bias)


@case("other values of eps: 1e-6, 1e-3, and 1e-2")
def eps_values(ctx):
    x = ctx.randn(4, 8, scale=0.05)
    weight, bias = params(ctx, 8)
    for eps in (1e-6, 1e-3, 1e-2):
        run(ctx, x, weight, bias, eps=eps)
        run(ctx, x, weight, bias, eps)


@case("rows on very different scales in one tensor, from 1e-4 to 50")
def mixed_scales(ctx):
    scales = np.array([[1e-4], [1e-2], [1.0], [10.0], [25.0], [50.0]], dtype=np.float32)
    x = np.clip(ctx.randn(6, 6) * scales, -100.0, 100.0)
    run(ctx, x, *params(ctx, 6))


@case("a single vector, a 4-D tensor, and square inputs")
def shapes(ctx):
    run(ctx, ctx.randn(4, scale=3.0), *params(ctx, 4))
    run(ctx, ctx.randn(2, 3, 4, 8, scale=4.0), *params(ctx, 8))
    x = ctx.randn(6, 6) * ctx.uniform(6, 1, low=0.2, high=20.0) + ctx.uniform(6, 1, low=-3, high=3)
    run(
        ctx,
        x,
        np.linspace(-1.0, 2.0, 6, dtype=np.float32),
        np.linspace(1.0, -1.0, 6, dtype=np.float32),
    )


# What LayerNorm means, without the oracle. Rejects no_mean_subtraction,
# uncentered_numerator, unbiased_variance, and normalizes_over_sequence.
@case("with unit gain and zero bias, every output vector has mean 0 and variance 1")
def zero_mean_unit_variance(ctx):
    def prop(rng):
        x = ctx.randn(2, 5, 16, scale=float(rng.uniform(0.5, 5.0)))
        x = x + ctx.uniform(2, 5, 1, low=-4.0, high=4.0)
        y = ctx.call("layer_norm", x, *plain(16)).astype(np.float64)
        check.shape(y, x.shape, label="layer_norm")
        check.finite(y, label="layer_norm")
        mean = y.mean(axis=-1)
        check.allclose(mean, np.zeros_like(mean), atol=1e-4, label="mean of each output vector")
        # The variance of each input vector is at least 0.02 here, so eps = 1e-5 moves
        # the output variance by well under 0.1%.
        var = np.mean(np.square(y - mean[..., None]), axis=-1)
        check.allclose(var, np.ones_like(var), rtol=2e-3, label="variance of each output vector")

    check.forall(prop, trials=8)


@case("adding a constant to a vector, or scaling it by 4, does not change the output")
def shift_and_scale_invariance(ctx):
    def prop(rng):
        x = ctx.randn(2, 5, 16, scale=float(rng.uniform(0.5, 3.0)))
        weight, bias = params(ctx, 16)
        y = ctx.call("layer_norm", x, weight, bias)
        shift = ctx.uniform(2, 5, 1, low=-8.0, high=8.0)
        y_shifted = ctx.call("layer_norm", x + shift, weight, bias)
        check.allclose(y_shifted, y, rtol=1e-3, atol=1e-4, label="layer_norm of the shifted input")
        y_scaled = ctx.call("layer_norm", x * np.float32(4.0), weight, bias)
        check.allclose(y_scaled, y, rtol=1e-3, atol=1e-4, label="layer_norm of the scaled input")

    check.forall(prop, trials=8)


@case("random shapes, scales, offsets, gains, biases, and eps")
def random_sweep(ctx):
    for shape in ((4, 8), (2, 5, 16), (6, 4), (2, 3, 4, 8)):
        scale = float(10.0 ** ctx.rng.uniform(-2.5, 0.7))
        offset = float(ctx.rng.uniform(-5.0, 5.0)) * scale
        eps = float(10.0 ** ctx.rng.uniform(-7, -3))
        x = ctx.randn(*shape, scale=scale, loc=offset)
        run(ctx, x, *params(ctx, shape[-1]), eps=eps)
