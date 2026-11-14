"""Tests for last-position-head.

Numeric cases compare the solution with the oracle. The speed case times it against
the reference of the same framework on a 512-token prompt: projecting every position
before slicing does 512 times the arithmetic. In wall-clock time that measures about
100x in PyTorch, 20x in JAX, and 8x in MLX, where the fixed cost of a GPU call hides
most of the difference; a correct solution measures about 1x in each.
"""

import numpy as np

from whetstone.judge import case, check

FN = "next_token_logits"
TOL = {"rtol": 1e-4, "atol": 1e-5}  # the tolerance stated in Constraints


def inputs(ctx, b, t, d, v):
    return ctx.randn(b, t, d), ctx.randn(v, d, scale=1.0 / np.sqrt(d))


@case("the worked example: B = 2, T = 2, D = 2, V = 4", public=True)
def worked_example(ctx):
    hidden = np.array([[[1.0, 2.0], [3.0, -1.0]], [[0.0, 1.0], [0.5, 0.5]]], dtype=np.float32)
    w_out = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, -1.0]], dtype=np.float32)
    got = ctx.check(FN, hidden, w_out, **TOL)
    expected = np.array([[3.0, -1.0, 2.0, 7.0], [0.5, 0.5, 1.0, 0.5]], dtype=np.float32)
    check.allclose(got, expected, atol=1e-5, label=FN)


@case("a random batch: B = 3, T = 5, D = 8, V = 11", public=True)
def random_batch(ctx):
    ctx.check(FN, *inputs(ctx, 3, 5, 8, 11), **TOL)


# Rejects first_position, second_to_last_position, and flattens_batch_and_time with a
# message that names the cause: only the last position of each sequence is nonzero.
@case("only the last position matters: every other hidden state is zero", public=True)
def only_last_position(ctx):
    hidden, w_out = inputs(ctx, 4, 6, 8, 11)
    hidden[:, :-1] = 0.0
    got = ctx.check(FN, hidden, w_out, **TOL)
    check.allclose(got, hidden[:, -1] @ w_out.T, atol=1e-5, label="logits of the last position")


# Rejects projects_every_position.
@case("speed: within 5x of the reference for B = 8, T = 512, D = 64, V = 8192", public=True)
def fast_enough(ctx):
    hidden, w_out = inputs(ctx, 8, 512, 64, 8192)
    ctx.check_speed(FN, hidden, w_out, max_ratio=5.0)


@case("a single position, and a batch of one")
def single_position(ctx):
    ctx.check(FN, *inputs(ctx, 3, 1, 8, 11), **TOL)
    ctx.check(FN, *inputs(ctx, 1, 7, 8, 11), **TOL)
    ctx.check(FN, *inputs(ctx, 1, 1, 4, 1), **TOL)


# B = T: hidden[-1] has the right shape here and must fail on the values.
@case("as many sequences as positions: B = T = 5")
def batch_equals_time(ctx):
    ctx.check(FN, *inputs(ctx, 5, 5, 8, 11), **TOL)


# V = D: an untransposed product runs here and must fail on the values.
@case("a square output matrix: V = D = 6, and B = T = D = V = 4")
def square_weights(ctx):
    ctx.check(FN, *inputs(ctx, 3, 5, 6, 6), **TOL)
    ctx.check(FN, *inputs(ctx, 4, 4, 4, 4), **TOL)


@case("the same answer at prompt scale: B = 2, T = 512, D = 64, V = 300")
def long_prompt(ctx):
    ctx.check(FN, *inputs(ctx, 2, 512, 64, 300), **TOL)


@case("positions are not interchangeable: reversing the sequence changes the answer")
def reversed_sequence(ctx):
    hidden, w_out = inputs(ctx, 3, 5, 8, 11)
    ctx.check(FN, hidden, w_out, **TOL)
    ctx.check(FN, hidden[:, ::-1].copy(), w_out, **TOL)
    ctx.check(FN, hidden[::-1].copy(), w_out, **TOL)


@case("random sizes for B, T, D, and V")
def random_sizes(ctx):
    for _ in range(3):
        b, t, d = (int(n) for n in ctx.rng.integers(1, 9, size=3))
        v = int(ctx.rng.integers(1, 40))
        ctx.check(FN, *inputs(ctx, b, t, d, v), **TOL)
