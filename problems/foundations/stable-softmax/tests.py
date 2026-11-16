"""Tests for stable-softmax. Every case compares the solution with the oracle."""

import numpy as np

from learnai.judge import case

BOTH = ("softmax", "log_softmax")


@case("the worked example: logits [1, 2, 3]", public=True)
def worked_example(ctx):
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    for fn in BOTH:
        ctx.check(fn, x)


@case("a random batch of logits, last axis", public=True)
def random_batch(ctx):
    x = ctx.randn(4, 8, scale=3.0)
    for fn in BOTH:
        ctx.check(fn, x, axis=-1)


@case("large logits: [1001, 1002, 1003] and a row spanning -1e4 to 1e4", public=True)
def large_logits(ctx):
    x = np.array([[1001.0, 1002.0, 1003.0], [1e4, 0.0, -1e4]], dtype=np.float32)
    for fn in BOTH:
        ctx.check(fn, x, axis=-1)


@case("the axis argument: a (3, 4, 5) tensor along each axis", public=True)
def each_axis(ctx):
    x = ctx.randn(3, 4, 5, scale=2.0)
    for fn in BOTH:
        for axis in (0, 1, 2, -2):
            ctx.check(fn, x, axis=axis)


@case("very negative logits")
def very_negative(ctx):
    x = np.array([[-1001.0, -1002.0, -1003.0], [-9000.0, -9000.5, -1e4]], dtype=np.float32)
    for fn in BOTH:
        ctx.check(fn, x, axis=-1)


@case("log_softmax of probabilities too small for float32")
def tiny_probabilities(ctx):
    x = np.array([[0.0, -200.0, -1000.0], [50.0, 20.0, -75.0]], dtype=np.float32)
    ctx.check("log_softmax", x, axis=-1)
    ctx.check("softmax", x, axis=-1)


@case("slices on very different scales in one tensor")
def mixed_scales(ctx):
    x = np.array([[0.0, 1.0, 2.0], [-500.0, -499.0, -498.0], [700.0, 700.0, 699.0]], np.float32)
    for fn in BOTH:
        ctx.check(fn, x, axis=1)
        ctx.check(fn, x.T.copy(), axis=0)


@case("square inputs along both axes")
def square_inputs(ctx):
    x = ctx.randn(5, 5, scale=4.0)
    for fn in BOTH:
        ctx.check(fn, x, axis=0)
        ctx.check(fn, x, axis=1)


@case("masked logits (-inf)")
def masked_logits(ctx):
    x = ctx.randn(3, 6, scale=2.0)
    x[0, 1] = x[0, 4] = x[2, 0] = -np.inf
    x[1, :5] = -np.inf  # a single surviving token gets probability 1
    for fn in BOTH:
        ctx.check(fn, x, axis=-1)


@case("an axis of length 1, and a single logit")
def length_one(ctx):
    for fn in BOTH:
        ctx.check(fn, ctx.randn(3, 1, scale=50.0), axis=-1)
        ctx.check(fn, np.array([-123.0], dtype=np.float32))


@case("random shapes, axes, and scales")
def random_shapes(ctx):
    for _ in range(8):
        shape = tuple(int(n) for n in ctx.rng.integers(1, 7, size=ctx.rng.integers(1, 5)))
        axis = int(ctx.rng.integers(-len(shape), len(shape)))
        scale = float(10.0 ** ctx.rng.uniform(-1, 3.5))
        x = ctx.randn(*shape, scale=scale)
        for fn in BOTH:
            ctx.check(fn, x, axis=axis)
