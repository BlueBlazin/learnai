"""Tests for broadcast-align.

Every case names the function it tests. The shapes are the test: sizes that all differ
(a wrong alignment raises or has the wrong shape), sizes that are equal (a wrong
alignment runs and returns wrong values), and sizes of 1.
"""

import numpy as np

from learnai.judge import case, check

TOL = {"rtol": 1e-4, "atol": 1e-4}  # the tolerance stated in Constraints


def labeled(batch, seq, width):
    """x[b, t, d] = 100 b + 10 t + d: every entry names its own position."""
    b, t, d = np.meshgrid(np.arange(batch), np.arange(seq), np.arange(width), indexing="ij")
    return (100 * b + 10 * t + d).astype(np.float32)


def factors(n):
    """1, 2, ..., n: no two rows share a factor, and no factor is 1 after the first."""
    return np.arange(1, n + 1, dtype=np.float32)


def points(ctx, n, dim):
    return ctx.uniform(n, dim, low=-3.0, high=3.0)


# ---------------------------------------------------------------- relative_positions


# Rejects positions_j_minus_i.
@case("relative_positions: the worked example, T = 3", public=True)
def positions_example(ctx):
    got = ctx.check("relative_positions", 3)
    check.equal(got, np.array([[0, -1, -2], [1, 0, -1], [2, 1, 0]]), label="relative_positions")


@case("relative_positions: T = 1 and T = 6")
def positions_sizes(ctx):
    ctx.check("relative_positions", 1)
    got = ctx.check("relative_positions", 6)
    check.that(
        got[5, 0] == 5 and got[0, 5] == -5,
        f"relative_positions: out[5, 0] is {got[5, 0]} and out[0, 5] is {got[0, 5]} for T = 6, "
        "expected 5 and -5",
    )


# ------------------------------------------------------------------------ scale_rows


@case("scale_rows: the worked example, x of shape (3, 1, 2)", public=True)
def scale_example(ctx):
    x = np.array([[[1.0, 2.0]], [[3.0, 4.0]], [[5.0, 6.0]]], dtype=np.float32)
    s = np.array([1.0, 10.0, 100.0], dtype=np.float32)
    got = ctx.check("scale_rows", x, s, **TOL)
    expected = np.array([[[1.0, 2.0]], [[30.0, 40.0]], [[500.0, 600.0]]], dtype=np.float32)
    check.allclose(got, expected, label="scale_rows", **TOL)


@case("scale_rows: all sizes differ, B = 2, T = 3, D = 4", public=True)
def scale_distinct(ctx):
    ctx.check("scale_rows", labeled(2, 3, 4), factors(2), **TOL)
    ctx.check("scale_rows", ctx.randn(2, 3, 4), ctx.randn(2), **TOL)


# Rejects scale_without_axes with wrong values: (B,) lines up with D and nothing raises.
@case("scale_rows: as many rows as features, B = D = 4 and T = 3", public=True)
def scale_batch_equals_width(ctx):
    ctx.check("scale_rows", labeled(4, 3, 4), factors(4), **TOL)
    ctx.check("scale_rows", ctx.randn(4, 3, 4), ctx.randn(4), **TOL)


# Rejects scale_on_time_axis with wrong values: (1, B, 1) lines up with T.
@case("scale_rows: as many rows as steps, B = T = 3 and D = 5")
def scale_batch_equals_steps(ctx):
    ctx.check("scale_rows", labeled(3, 3, 5), factors(3), **TOL)
    ctx.check("scale_rows", ctx.randn(3, 3, 5), ctx.randn(3), **TOL)


# Every placement of s has the right shape here; only the values tell them apart.
@case("scale_rows: all axes the same size, B = T = D = 3")
def scale_all_equal(ctx):
    ctx.check("scale_rows", labeled(3, 3, 3), factors(3), **TOL)
    ctx.check("scale_rows", ctx.randn(3, 3, 3), ctx.randn(3), **TOL)


@case("scale_rows: a batch of one, a single feature, and all sizes 1")
def scale_size_one(ctx):
    ctx.check("scale_rows", ctx.randn(1, 3, 4), np.array([-2.0], dtype=np.float32), **TOL)
    ctx.check("scale_rows", ctx.randn(4, 3, 1), factors(4), **TOL)
    ctx.check("scale_rows", ctx.randn(1, 1, 1), np.array([3.0], dtype=np.float32), **TOL)


# ------------------------------------------------------------------ pairwise_sq_dist


@case("pairwise_sq_dist: the worked example, 3 points against 4 in the plane", public=True)
def dist_example(ctx):
    a = np.array([[0.0, 0.0], [3.0, 4.0], [6.0, 8.0]], dtype=np.float32)
    b = np.array([[0.0, 0.0], [3.0, 0.0], [0.0, 4.0], [6.0, 8.0]], dtype=np.float32)
    got = ctx.check("pairwise_sq_dist", a, b, **TOL)
    expected = np.array(
        [[0.0, 9.0, 16.0, 100.0], [25.0, 16.0, 9.0, 25.0], [100.0, 73.0, 52.0, 0.0]],
        dtype=np.float32,
    )
    check.allclose(got, expected, label="pairwise_sq_dist", **TOL)


# Rejects elementwise_distance (shape (5,)) and distance_table_transposed (wrong values:
# the table of two different point sets is not symmetric).
@case("pairwise_sq_dist: as many points in a as in b, N = M = 5 and D = 3", public=True)
def dist_same_count(ctx):
    ctx.check("pairwise_sq_dist", points(ctx, 5, 3), points(ctx, 5, 3), **TOL)


# Rejects distance_sums_wrong_axis with wrong values: (N, D) has the shape of (N, M).
@case("pairwise_sq_dist: all sizes equal, N = M = D = 4")
def dist_all_equal(ctx):
    ctx.check("pairwise_sq_dist", points(ctx, 4, 4), points(ctx, 4, 4), **TOL)


@case("pairwise_sq_dist: a set against itself has a zero diagonal")
def dist_self(ctx):
    a = points(ctx, 5, 3)
    got = ctx.check("pairwise_sq_dist", a, a, **TOL)
    check.allclose(np.diag(got), np.zeros(5, dtype=np.float32), label="the diagonal", **TOL)


@case("pairwise_sq_dist: a single point on either side, one feature, and all sizes 1")
def dist_size_one(ctx):
    ctx.check("pairwise_sq_dist", points(ctx, 1, 3), points(ctx, 5, 3), **TOL)
    ctx.check("pairwise_sq_dist", points(ctx, 5, 3), points(ctx, 1, 3), **TOL)
    ctx.check("pairwise_sq_dist", points(ctx, 5, 1), points(ctx, 3, 1), **TOL)
    ctx.check("pairwise_sq_dist", points(ctx, 1, 1), points(ctx, 1, 1), **TOL)


@case("pairwise_sq_dist: more points than features and the reverse, values up to 10")
def dist_wide(ctx):
    ctx.check("pairwise_sq_dist", ctx.uniform(3, 4, low=-10, high=10), points(ctx, 2, 4), **TOL)
    ctx.check("pairwise_sq_dist", points(ctx, 2, 16), ctx.uniform(3, 16, low=-10, high=10), **TOL)
