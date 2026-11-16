"""Tests for swiglu-mlp. Every case compares the solution with the oracle.

The widths d, h, and d_out differ in most cases, because weights applied without the
transpose only run at all when the three are equal; one case then makes them equal on
purpose, so that mistake fails on values instead of on shapes.
"""

import numpy as np

from learnai.judge import case, check

FN = "swiglu"
TOL = {"rtol": 1e-4, "atol": 1e-5}  # the tolerance stated in Constraints


def weights(ctx, d, h, d_out, scale=1.0):
    """Weights in (out, in) layout, scaled so that g and u are of order `scale`."""
    s = scale / np.sqrt(d)
    return (
        ctx.randn(h, d, scale=s),
        ctx.randn(h, d, scale=s),
        ctx.randn(d_out, h, scale=1.0 / np.sqrt(h)),
    )


@case("the worked example: d = 2, h = 3, d_out = 2", public=True)
def worked_example(ctx):
    x = np.array([1.0, 2.0], dtype=np.float32)
    w_gate = np.array([[1.0, 0.0], [0.0, -1.0], [1.0, 1.0]], dtype=np.float32)
    w_up = np.array([[1.0, 1.0], [2.0, 0.0], [0.0, 0.5]], dtype=np.float32)
    w_down = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, -1.0]], dtype=np.float32)
    got = ctx.check(FN, x, w_gate, w_up, w_down, **TOL)
    expected = np.array([5.050899, -3.334534], dtype=np.float32)
    check.allclose(got, expected, atol=1e-5, label=FN)


@case("a random batch: x of shape (2, 5, 8), h = 16, d_out = 8", public=True)
def random_batch(ctx):
    ctx.check(FN, ctx.randn(2, 5, 8), *weights(ctx, 8, 16, 8), **TOL)


# Rejects weights_not_transposed: no two of the widths agree.
@case("three different widths: d = 6, h = 10, d_out = 4", public=True)
def three_widths(ctx):
    ctx.check(FN, ctx.randn(3, 4, 6), *weights(ctx, 6, 10, 4), **TOL)


# The branches are told apart by sign: the gate branch is negative everywhere, so
# silu(g) is small and negative, while the up branch is positive and large. Rejects
# activation_on_up_branch, sigmoid_gate, and gelu_instead_of_silu by a wide margin.
@case("a negative gate branch and a positive up branch", public=True)
def negative_gate(ctx):
    x = np.abs(ctx.randn(4, 6)) + 0.5
    w_gate = -np.abs(ctx.randn(9, 6, scale=0.3))
    w_up = np.abs(ctx.randn(9, 6, scale=0.3))
    w_down = ctx.randn(5, 9, scale=0.3)
    ctx.check(FN, x, w_gate, w_up, w_down, **TOL)


# Square weights: x @ W runs without an error here, so it has to fail on the values.
@case("all widths equal: d = h = d_out = 5, and a (5, 5) input")
def square_weights(ctx):
    ctx.check(FN, ctx.randn(5, 5), *weights(ctx, 5, 5, 5), **TOL)
    ctx.check(FN, ctx.randn(2, 5, 5, scale=2.0), *weights(ctx, 5, 5, 5), **TOL)


@case("a single vector, a 2-D batch, and a single hidden unit")
def input_ranks(ctx):
    ctx.check(FN, ctx.randn(8), *weights(ctx, 8, 12, 8), **TOL)
    ctx.check(FN, ctx.randn(7, 8), *weights(ctx, 8, 12, 3), **TOL)
    ctx.check(FN, ctx.randn(2, 3, 4), *weights(ctx, 4, 1, 6), **TOL)
    ctx.check(FN, ctx.randn(3, 1), *weights(ctx, 1, 7, 1), **TOL)


@case("saturated gates: pre-activations between -100 and 100")
def saturated_gates(ctx):
    # One-hot inputs select columns of the weights, so g and u are exactly the values
    # written here: gates from fully closed (-100) to fully open (100).
    g = np.array([-100.0, -90.0, -40.0, -10.0, 0.0, 10.0, 40.0, 90.0, 100.0], dtype=np.float32)
    w_gate = np.stack([g, g[::-1], 0.5 * g], axis=1)  # (9, 3)
    w_up = ctx.uniform(9, 3, low=-100.0, high=100.0)
    w_down = ctx.randn(4, 9, scale=0.3)
    got = ctx.check(FN, np.eye(3, dtype=np.float32), w_gate, w_up, w_down, **TOL)
    check.finite(got, label=FN)


@case("a zero input gives a zero output, and gates near zero")
def small_inputs(ctx):
    w = weights(ctx, 6, 10, 4)
    ctx.check(FN, np.zeros((2, 6), dtype=np.float32), *w, **TOL)
    # silu(z) is about z / 2 near zero, so the output is quadratic in a small input.
    ctx.check(FN, ctx.randn(4, 6, scale=1e-2), *w, rtol=1e-4, atol=1e-8)


@case("the output is not symmetric in the two branches: swapping w_gate and w_up changes it")
def swapped_branches(ctx):
    x = ctx.randn(3, 4, 8)
    w_gate, w_up, w_down = weights(ctx, 8, 12, 8, scale=2.0)
    ctx.check(FN, x, w_gate, w_up, w_down, **TOL)
    ctx.check(FN, x, w_up, w_gate, w_down, **TOL)


@case("random shapes, widths, and scales")
def random_sweep(ctx):
    for _ in range(4):
        d, h, d_out = (int(n) for n in ctx.rng.integers(1, 17, size=3))
        lead = tuple(int(n) for n in ctx.rng.integers(1, 5, size=ctx.rng.integers(0, 3)))
        scale = float(10.0 ** ctx.rng.uniform(-1, 1))
        ctx.check(FN, ctx.randn(*lead, d), *weights(ctx, d, h, d_out, scale=scale), **TOL)
