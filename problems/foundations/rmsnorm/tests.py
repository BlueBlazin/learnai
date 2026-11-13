"""Tests for rmsnorm. Numeric cases compare with the oracle; one property case states
what the function means without it."""

import numpy as np

from whetstone.judge import case, check


def gain(ctx, d):
    """A non-trivial gain vector: positive and negative entries of order 1."""
    return ctx.uniform(d, low=0.5, high=1.5) * np.where(ctx.rng.random(d) < 0.25, -1, 1).astype(
        np.float32
    )


@case("the worked example: x = [3, 4], weight = [1, 0.5]", public=True)
def worked_example(ctx):
    x = np.array([3.0, 4.0], dtype=np.float32)
    weight = np.array([1.0, 0.5], dtype=np.float32)
    got = ctx.check("rms_norm", x, weight)
    check.allclose(got, np.array([0.848528, 0.565685], np.float32), label="rms_norm")


@case("a random batch of vectors, shape (4, 8)", public=True)
def random_batch(ctx):
    ctx.check("rms_norm", ctx.randn(4, 8), gain(ctx, 8))


# Rejects normalizes_over_sequence: the tokens have very different magnitudes.
@case("batch and sequence axes: shape (2, 5, 16), each token normalized on its own", public=True)
def batch_and_sequence(ctx):
    x = ctx.randn(2, 5, 16) * ctx.uniform(2, 5, 1, low=0.1, high=20.0)
    ctx.check("rms_norm", x, gain(ctx, 16))


# Rejects eps_outside_sqrt and no_eps: here mean(x^2) is comparable to eps.
@case("small activations: rows of size 1e-3 to 1e-2, where eps matters", public=True)
def small_activations(ctx):
    scales = np.array([[1e-3], [3e-3], [1e-2], [3e-2]], dtype=np.float32)
    x = ctx.randn(4, 8) * scales
    ctx.check("rms_norm", x, gain(ctx, 8))


# Rejects square_overflows.
@case("large activations: rows of size 1, 1e15, 1e20, and 1e28", public=True)
def large_activations(ctx):
    scales = np.array([[1.0], [1e15], [1e20], [1e28]], dtype=np.float32)
    x = ctx.randn(4, 8) * scales
    got = ctx.check("rms_norm", x, gain(ctx, 8))
    check.finite(got, label="rms_norm")


@case("all-zero vectors come out as zeros")
def zero_vectors(ctx):
    x = ctx.randn(4, 8)
    x[1] = 0.0
    got = ctx.check("rms_norm", x, gain(ctx, 8))
    check.finite(got, label="rms_norm")
    ctx.check("rms_norm", np.zeros((2, 5, 16), dtype=np.float32), gain(ctx, 16))


@case("other values of eps: 1e-6, 1e-3, and 1e-2")
def eps_values(ctx):
    x = ctx.randn(4, 8, scale=0.05)
    weight = gain(ctx, 8)
    for eps in (1e-6, 1e-3, 1e-2):
        ctx.check("rms_norm", x, weight, eps=eps)
        ctx.check("rms_norm", x, weight, eps)


@case("rows on very different scales in one tensor, from 1e-4 to 1e29")
def mixed_scales(ctx):
    scales = np.array([[1e-4], [1.0], [1e9], [1e19], [3e19], [1e29]], dtype=np.float32)
    x = ctx.randn(6, 6) * scales
    ctx.check("rms_norm", x, gain(ctx, 6))


# Rejects subtracts_mean even when a learner's bug only shows on shifted data.
@case("activations with a large mean are not centered")
def not_centered(ctx):
    ctx.check("rms_norm", ctx.randn(2, 5, 16, loc=5.0), gain(ctx, 16))
    ctx.check("rms_norm", np.full((4, 8), -2.5, dtype=np.float32), gain(ctx, 8))


@case("a last axis of length 1, a single vector, and a 4-D tensor")
def shapes(ctx):
    ctx.check("rms_norm", ctx.randn(5, 1, scale=3.0), np.array([1.5], dtype=np.float32))
    ctx.check("rms_norm", ctx.randn(12), gain(ctx, 12))
    ctx.check("rms_norm", ctx.randn(2, 3, 4, 8, scale=4.0), gain(ctx, 8))


@case("square inputs with a non-uniform gain")
def square_inputs(ctx):
    x = ctx.randn(6, 6) * ctx.uniform(6, 1, low=0.2, high=30.0)
    ctx.check("rms_norm", x, np.linspace(-1.0, 2.0, 6, dtype=np.float32))
    x3 = ctx.randn(4, 4, 4) * ctx.uniform(4, 4, 1, low=0.2, high=30.0)
    ctx.check("rms_norm", x3, np.linspace(0.5, 2.0, 4, dtype=np.float32))


@case("with unit gain the output has root mean square 1, whatever the input scale")
def unit_rms(ctx):
    def prop(rng):
        scale = float(10.0 ** rng.uniform(-1, 28))
        x = ctx.randn(2, 5, 16, scale=scale)
        y = ctx.call("rms_norm", x, np.ones(16, dtype=np.float32))
        check.shape(y, x.shape, label="rms_norm")
        check.finite(y, label="rms_norm")
        rms = np.sqrt(np.mean(np.square(y.astype(np.float64)), axis=-1))
        # mean(x^2) is at least 1e-3 here, so eps = 1e-5 moves the RMS by under 1%.
        check.allclose(rms, np.ones_like(rms), rtol=1e-2, label="root mean square of the output")
        # Rescaling the input by a power of two leaves the output unchanged.
        if scale > 1e3:
            y2 = ctx.call("rms_norm", x * np.float32(0.25), np.ones(16, dtype=np.float32))
            check.allclose(y2, y, label="rms_norm of the rescaled input")

    check.forall(prop, trials=8)


@case("random shapes, scales, gains, and eps")
def random_sweep(ctx):
    for _ in range(3):
        shape = tuple(int(n) for n in ctx.rng.integers(1, 7, size=ctx.rng.integers(1, 5)))
        scale = float(10.0 ** ctx.rng.uniform(-2.5, 25))
        eps = float(10.0 ** ctx.rng.uniform(-7, -3))
        ctx.check("rms_norm", ctx.randn(*shape, scale=scale), gain(ctx, shape[-1]), eps=eps)
