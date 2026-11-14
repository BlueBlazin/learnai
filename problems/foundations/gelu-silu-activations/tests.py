"""Tests for gelu-silu-activations. Every case compares the solution with the oracle."""

import numpy as np

from whetstone.judge import case, check

TOL = {"rtol": 1e-4, "atol": 1e-5}  # the tolerance stated in Constraints


def check_all(ctx, x):
    """Both GELU forms and SiLU on the same input."""
    ctx.check("gelu", x, approximate="none", **TOL)
    ctx.check("gelu", x, approximate="tanh", **TOL)
    return ctx.check("silu", x, **TOL)


@case("the worked example: x = [-3, -1, 0, 1, 3]", public=True)
def worked_example(ctx):
    x = np.array([-3.0, -1.0, 0.0, 1.0, 3.0], dtype=np.float32)
    exact = ctx.check("gelu", x, **TOL)
    tanh = ctx.check("gelu", x, approximate="tanh", **TOL)
    swish = ctx.check("silu", x, **TOL)
    expected = {
        'gelu(x, "none")': (exact, [-0.004050, -0.158655, 0.0, 0.841345, 2.995950]),
        'gelu(x, "tanh")': (tanh, [-0.003637, -0.158808, 0.0, 0.841192, 2.996363]),
        "silu(x)": (swish, [-0.142278, -0.268941, 0.0, 0.731059, 2.857723]),
    }
    for label, (got, want) in expected.items():
        check.allclose(got, np.array(want, dtype=np.float32), label=label, **TOL)


@case("a random batch, shape (4, 8)", public=True)
def random_batch(ctx):
    check_all(ctx, ctx.randn(4, 8, scale=2.0))


# Rejects tanh_for_exact and exact_for_tanh: the two forms differ by up to 4.7e-4 here,
# and the outputs for negative x are small enough that the difference is far outside
# the tolerance. Also rejects wrong_constant, erf_without_sqrt2, and quick_gelu_for_tanh.
@case("the two GELU forms are different functions: a grid from -4 to 4", public=True)
def grid(ctx):
    x = np.linspace(-4.0, 4.0, 33, dtype=np.float32)
    exact = ctx.check("gelu", x, approximate="none", **TOL)
    tanh = ctx.check("gelu", x, approximate="tanh", **TOL)
    check.that(
        float(np.abs(exact - tanh).max()) > 1e-4,
        'gelu: approximate="none" and approximate="tanh" returned the same values',
    )


# Rejects sigmoid_overflows: exp(x) is inf in float32 from x = 88.73 on.
@case("large magnitudes: |x| from 30 to 1e4, both signs", public=True)
def large_magnitudes(ctx):
    half = np.array([30.0, 88.0, 89.0, 100.0, 750.0, 1e3, 1e4], dtype=np.float32)
    x = np.concatenate([-half[::-1], half])
    got = check_all(ctx, x)
    check.finite(got, label="silu")


@case("approximate passed by position, and left at its default")
def argument_passing(ctx):
    x = ctx.randn(4, 8, scale=2.5)
    ctx.check("gelu", x, "tanh", **TOL)
    ctx.check("gelu", x, "none", **TOL)
    default = ctx.call("gelu", x)
    check.allclose(default, ctx.expected("gelu", x, "none"), label="gelu(x)", **TOL)


@case("the negative tail: x from -8 to -1, where the outputs are small")
def negative_tail(ctx):
    check_all(ctx, np.linspace(-8.0, -1.0, 33, dtype=np.float32))


@case("small inputs around zero, and exact zeros")
def near_zero(ctx):
    x = ctx.randn(4, 8, scale=0.05)
    x[0, 0] = x[2, 3] = 0.0
    x[3] = 0.0
    check_all(ctx, x)


@case("silu on both sides of the float32 overflow point of exp, x = +-88.7")
def exp_overflow_point(ctx):
    x = np.array([[-90.0, -88.8, -88.7, -87.0], [87.0, 88.7, 88.8, 90.0]], dtype=np.float32)
    got = ctx.check("silu", x, **TOL)
    check.finite(got, label="silu")
    ctx.check("gelu", x, **TOL)


@case("a single value, a 1-D vector, and a 4-D tensor")
def shapes(ctx):
    check_all(ctx, np.array([1.5], dtype=np.float32))
    check_all(ctx, ctx.randn(33, scale=3.0))
    check_all(ctx, ctx.randn(2, 3, 4, 5, scale=2.0))


@case("values on mixed scales in one tensor, from 1e-3 to 1e4")
def mixed_scales(ctx):
    scales = np.array([[1e-3], [1.0], [30.0], [1e4]], dtype=np.float32)
    check_all(ctx, ctx.randn(4, 8) * scales / 3.0)


@case("random shapes and scales")
def random_sweep(ctx):
    for _ in range(2):
        shape = tuple(int(n) for n in ctx.rng.integers(1, 7, size=ctx.rng.integers(1, 5)))
        scale = float(10.0 ** ctx.rng.uniform(-1, 2))
        check_all(ctx, ctx.randn(*shape, scale=scale))
