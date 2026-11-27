"""Tests for size-one-axes.

Every case names the function it tests. Each function is run on shapes with an axis of
size 1 (where a bare squeeze removes too much), on shapes with equal axes (where a sum
that lost its axis still broadcasts), and on shapes whose sizes all differ.
"""

import numpy as np

from learnai.judge import case, check

EXACT = {"rtol": 0.0, "atol": 0.0}  # last_step and column copy values


def labeled(batch, seq, width):
    """x[b, t, d] = 100 b + 10 t + d: every entry names its own position."""
    b, t, d = np.meshgrid(np.arange(batch), np.arange(seq), np.arange(width), indexing="ij")
    return (100 * b + 10 * t + d).astype(np.float32)


def table(rows, cols):
    """x[b, v] = 10 b + v."""
    return (10 * np.arange(rows)[:, None] + np.arange(cols)[None, :]).astype(np.float32)


def positive(ctx, *shape):
    return ctx.uniform(*shape, low=0.5, high=2.0)


def check_normalize(ctx, x, dim):
    """Compare with the oracle, then state the meaning: slices along dim sum to 1."""
    got = ctx.check("normalize_over", x, dim)
    sums = got.sum(axis=dim)
    check.allclose(sums, np.ones_like(sums), label=f"sums of normalize_over(x, {dim}) over dim")


# -------------------------------------------------------------------- normalize_over


@case("normalize_over: the worked example, a (2, 3) input over dim 1 and dim 0", public=True)
def normalize_example(ctx):
    x = np.array([[1.0, 1.0, 2.0], [3.0, 4.0, 1.0]], dtype=np.float32)
    got = ctx.check("normalize_over", x, 1)
    rows = np.array([[0.25, 0.25, 0.5], [0.375, 0.5, 0.125]], dtype=np.float32)
    check.allclose(got, rows, label="normalize_over(x, 1)")
    got = ctx.check("normalize_over", x, 0)
    cols = np.array([[0.25, 0.2, 2 / 3], [0.75, 0.8, 1 / 3]], dtype=np.float32)
    check.allclose(got, cols, label="normalize_over(x, 0)")


# Rejects no_keepdim with wrong values: the (4,) sum of rows lines up with the columns.
@case("normalize_over: a square (4, 4) input over dim 1 and dim 0", public=True)
def normalize_square(ctx):
    x = positive(ctx, 4, 4)
    check_normalize(ctx, x, 1)
    check_normalize(ctx, x, 0)


# Rejects positive_dim_only with wrong values, and no_keepdim again.
@case("normalize_over: negative dim on a square (4, 4) input", public=True)
def normalize_negative_dim(ctx):
    x = positive(ctx, 4, 4)
    check_normalize(ctx, x, -1)
    check_normalize(ctx, x, -2)


@case("normalize_over: one axis, and three axes of different sizes over each dim")
def normalize_ranks(ctx):
    v = positive(ctx, 5)
    check_normalize(ctx, v, 0)
    check_normalize(ctx, v, -1)
    x = positive(ctx, 2, 3, 4)
    for dim in (0, 1, 2, -2):
        check_normalize(ctx, x, dim)


@case("normalize_over: three and four axes of equal size, B = T = D = 3")
def normalize_equal_axes(ctx):
    x = positive(ctx, 3, 3, 3)
    for dim in (1, 2, -2, -3):
        check_normalize(ctx, x, dim)
    y = positive(ctx, 3, 3, 3, 3)
    for dim in (0, 2, -1, -3):
        check_normalize(ctx, y, dim)


@case("normalize_over: axes of size 1, summed over and not")
def normalize_size_one(ctx):
    check_normalize(ctx, positive(ctx, 1, 4), 0)  # every entry becomes 1
    check_normalize(ctx, positive(ctx, 1, 4), 1)
    check_normalize(ctx, positive(ctx, 4, 1), -1)
    check_normalize(ctx, positive(ctx, 1, 1, 1), 1)
    check_normalize(ctx, positive(ctx, 1), 0)


# ------------------------------------------------------------------------- last_step


@case("last_step: the worked example, x of shape (2, 3, 2)", public=True)
def last_step_example(ctx):
    got = ctx.check("last_step", labeled(2, 3, 2), **EXACT)
    expected = np.array([[20.0, 21.0], [120.0, 121.0]], dtype=np.float32)
    check.allclose(got, expected, label="last_step", **EXACT)


# The statement's second example. Rejects bare_squeeze: the result must be (1, 4), and
# squeeze() returns (4,).
@case("last_step: a batch of one keeps its batch axis, x of shape (1, 2, 4)", public=True)
def last_step_batch_of_one(ctx):
    got = ctx.check("last_step", labeled(1, 2, 4), **EXACT)
    expected = np.array([[10.0, 11.0, 12.0, 13.0]], dtype=np.float32)
    check.allclose(got, expected, label="last_step", **EXACT)


# Rejects bare_squeeze: the result must be (2, 1), and squeeze() returns (2,).
@case("last_step: a single feature keeps its feature axis, x of shape (2, 3, 1)", public=True)
def last_step_one_feature(ctx):
    ctx.check("last_step", labeled(2, 3, 1), **EXACT)


@case("last_step: a single step, x of shape (2, 1, 4)")
def last_step_one_step(ctx):
    ctx.check("last_step", labeled(2, 1, 4), **EXACT)


# The result must be (1, 1); a bare squeeze() returns a tensor with no axes at all.
@case("last_step: B = T = D = 1 returns shape (1, 1)")
def last_step_all_one(ctx):
    ctx.check("last_step", np.array([[[7.0]]], dtype=np.float32), **EXACT)


# Rejects last_feature_not_step with wrong values: (B, T) has the shape of (B, D).
@case("last_step: as many steps as features, T = D = 3, and B = T = D = 3")
def last_step_equal_axes(ctx):
    ctx.check("last_step", labeled(2, 3, 3), **EXACT)
    ctx.check("last_step", labeled(3, 3, 3), **EXACT)
    ctx.check("last_step", ctx.randn(3, 3, 3), **EXACT)


@case("last_step: all sizes differ, B = 4, T = 5, D = 3")
def last_step_distinct(ctx):
    ctx.check("last_step", ctx.randn(4, 5, 3), **EXACT)


# ---------------------------------------------------------------------------- column


# Rejects column_drops_axis: the result must be (2, 1), and x[:, j] is (2,).
@case("column: the worked example, column 1 of a (2, 3) input", public=True)
def column_example(ctx):
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    got = ctx.check("column", x, 1, **EXACT)
    check.allclose(got, np.array([[2.0], [5.0]], dtype=np.float32), label="column", **EXACT)


@case("column: the first and the last column of a (3, 5) input", public=True)
def column_ends(ctx):
    x = table(3, 5)
    ctx.check("column", x, 0, **EXACT)
    ctx.check("column", x, 4, **EXACT)


# Rejects column_reads_row with wrong values: (V, 1) has the shape of (B, 1).
@case("column: a square input, B = V = 4")
def column_square(ctx):
    x = table(4, 4)
    for j in (0, 2, 3):
        ctx.check("column", x, j, **EXACT)


@case("column: a batch of one, a single column, and a (1, 1) input")
def column_size_one(ctx):
    ctx.check("column", table(1, 5), 3, **EXACT)
    ctx.check("column", table(4, 1), 0, **EXACT)
    ctx.check("column", np.array([[7.0]], dtype=np.float32), 0, **EXACT)
