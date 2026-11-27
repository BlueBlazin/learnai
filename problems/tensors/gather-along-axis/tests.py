"""Tests for gather-along-axis.

Every case names the function it tests. Gathering copies values, so every comparison
is exact. Index tensors are passed as numpy int64 and arrive as the framework's default
integer type: int64 in PyTorch, int32 in JAX and MLX.
"""

import numpy as np

from learnai.judge import case, check

EXACT = {"rtol": 0.0, "atol": 0.0}


def ints(values):
    return np.array(values, dtype=np.int64)


def table(rows, cols):
    """values[b, v] = 10 b + v: every entry names its own row and column."""
    return (10 * np.arange(rows)[:, None] + np.arange(cols)[None, :]).astype(np.float32)


def labeled(batch, seq, width):
    """x[b, t, d] = 100 b + 10 t + d: every entry names its own position."""
    b, t, d = np.meshgrid(np.arange(batch), np.arange(seq), np.arange(width), indexing="ij")
    return (100 * b + 10 * t + d).astype(np.float32)


def perms(ctx, batch, seq):
    """One random permutation of 0 .. seq - 1 per row."""
    return np.stack([ctx.rng.permutation(seq) for _ in range(batch)]).astype(np.int64)


# ---------------------------------------------------------------------- token_scores


# Rejects scores_by_plain_indexing (shape (2, 3, 2, 3)) and scores_gather_on_time_axis
# (wrong values: with V <= T every id is also a valid position).
@case("token_scores: the worked example, logits of shape (2, 3, 2)", public=True)
def scores_example(ctx):
    ids = ints([[1, 0, 1], [0, 0, 1]])
    got = ctx.check("token_scores", labeled(2, 3, 2), ids, **EXACT)
    expected = np.array([[1.0, 10.0, 21.0], [100.0, 110.0, 121.0]], dtype=np.float32)
    check.allclose(got, expected, label="token_scores", **EXACT)


@case("token_scores: all sizes differ, B = 2, T = 3, V = 5", public=True)
def scores_distinct(ctx):
    ctx.check("token_scores", labeled(2, 3, 5), ints([[4, 0, 2], [3, 3, 1]]), **EXACT)
    ctx.check("token_scores", ctx.randn(2, 3, 5), ctx.randint(0, 5, (2, 3)), **EXACT)


# Rejects scores_bare_squeeze: the result must be (1, 3), and squeeze() returns (3,).
@case("token_scores: a batch of one keeps its batch axis, logits of shape (1, 3, 5)", public=True)
def scores_batch_of_one(ctx):
    ctx.check("token_scores", labeled(1, 3, 5), ints([[4, 0, 2]]), **EXACT)


# Rejects scores_bare_squeeze: the result must be (2, 1), and squeeze() returns (2,).
@case("token_scores: a single position keeps its axis, logits of shape (2, 1, 5)", public=True)
def scores_one_position(ctx):
    ctx.check("token_scores", labeled(2, 1, 5), ints([[3], [1]]), **EXACT)


@case("token_scores: as many positions as tokens, T = V = 4, and B = T = V = 3")
def scores_equal_axes(ctx):
    ctx.check("token_scores", labeled(2, 4, 4), ints([[3, 0, 2, 2], [1, 3, 0, 1]]), **EXACT)
    ctx.check("token_scores", labeled(3, 3, 3), ints([[2, 0, 1], [1, 1, 0], [0, 2, 2]]), **EXACT)
    ctx.check("token_scores", ctx.randn(3, 3, 3), ctx.randint(0, 3, (3, 3)), **EXACT)


@case("token_scores: a vocabulary of one token, and all sizes 1")
def scores_size_one(ctx):
    ctx.check("token_scores", labeled(2, 3, 1), np.zeros((2, 3), dtype=np.int64), **EXACT)
    ctx.check("token_scores", labeled(1, 1, 1), np.zeros((1, 1), dtype=np.int64), **EXACT)


# ---------------------------------------------------------------------- take_columns


@case("take_columns: the worked example, 3 picks from each row of a (2, 4) table", public=True)
def columns_example(ctx):
    got = ctx.check("take_columns", table(2, 4), ints([[3, 0, 3], [1, 2, 2]]), **EXACT)
    expected = np.array([[3.0, 0.0, 3.0], [11.0, 12.0, 12.0]], dtype=np.float32)
    check.allclose(got, expected, label="take_columns", **EXACT)


# Rejects columns_gather_on_batch_axis with wrong values: every index is a valid row.
@case("take_columns: more rows than columns, B = 5, V = 3, K = 2", public=True)
def columns_tall(ctx):
    idx = ints([[2, 0], [1, 1], [0, 2], [2, 2], [0, 1]])
    ctx.check("take_columns", table(5, 3), idx, **EXACT)


@case("take_columns: all sizes equal, B = V = K = 3")
def columns_all_equal(ctx):
    ctx.check("take_columns", table(3, 3), ints([[2, 0, 1], [1, 1, 0], [0, 2, 2]]), **EXACT)
    ctx.check("take_columns", ctx.randn(3, 3), ctx.randint(0, 3, (3, 3)), **EXACT)


@case("take_columns: more picks than columns, K = 6 from V = 4")
def columns_many_picks(ctx):
    ctx.check("take_columns", ctx.randn(2, 4), ctx.randint(0, 4, (2, 6)), **EXACT)


@case("take_columns: one pick per row, a batch of one, one column, and a (1, 1) table")
def columns_size_one(ctx):
    ctx.check("take_columns", table(5, 3), ints([[2], [0], [1], [1], [0]]), **EXACT)
    ctx.check("take_columns", table(1, 4), ints([[3, 0, 3]]), **EXACT)
    ctx.check("take_columns", table(5, 1), np.zeros((5, 2), dtype=np.int64), **EXACT)
    ctx.check("take_columns", table(1, 1), np.zeros((1, 1), dtype=np.int64), **EXACT)


# --------------------------------------------------------------------- reorder_steps


# Rejects reorder_expands_output: with an unexpanded index only feature 0 (the entries
# ending in 0) is right. Rejects reorder_unexpanded_index (shape (2, 3, 1)) and
# reorder_one_perm_for_all.
@case("reorder_steps: the worked example, x of shape (2, 3, 2)", public=True)
def reorder_example(ctx):
    perm = ints([[2, 0, 1], [0, 2, 1]])
    got = ctx.check("reorder_steps", labeled(2, 3, 2), perm, **EXACT)
    expected = np.array(
        [
            [[20.0, 21.0], [0.0, 1.0], [10.0, 11.0]],
            [[100.0, 101.0], [120.0, 121.0], [110.0, 111.0]],
        ],
        dtype=np.float32,
    )
    check.allclose(got, expected, label="reorder_steps", **EXACT)


# Rejects reorder_on_batch_axis with wrong values: every step number is a valid row.
@case("reorder_steps: more rows than steps, B = 4, T = 3, D = 5", public=True)
def reorder_tall(ctx):
    perm = ints([[2, 0, 1], [0, 1, 2], [1, 2, 0], [2, 1, 0]])
    ctx.check("reorder_steps", labeled(4, 3, 5), perm, **EXACT)


@case("reorder_steps: all sizes equal, B = T = D = 3")
def reorder_all_equal(ctx):
    perm = ints([[2, 0, 1], [1, 2, 0], [0, 2, 1]])
    ctx.check("reorder_steps", labeled(3, 3, 3), perm, **EXACT)
    ctx.check("reorder_steps", ctx.randn(3, 3, 3), perms(ctx, 3, 3), **EXACT)


@case("reorder_steps: the identity permutation returns x, and a reversal reverses it")
def reorder_identity(ctx):
    x = ctx.randn(2, 6, 3)
    forward = np.tile(np.arange(6, dtype=np.int64), (2, 1))
    got = ctx.check("reorder_steps", x, forward, **EXACT)
    check.allclose(got, x, label="reorder_steps with the identity", **EXACT)
    got = ctx.check("reorder_steps", x, forward[:, ::-1].copy(), **EXACT)
    check.allclose(got, x[:, ::-1], label="reorder_steps with a reversal", **EXACT)


@case("reorder_steps: random permutations, B = 2, T = 6, D = 3")
def reorder_random(ctx):
    for _ in range(3):
        ctx.check("reorder_steps", ctx.randn(2, 6, 3), perms(ctx, 2, 6), **EXACT)


@case("reorder_steps: a batch of one, a single step, a single feature, and all sizes 1")
def reorder_size_one(ctx):
    ctx.check("reorder_steps", labeled(1, 3, 5), ints([[1, 2, 0]]), **EXACT)
    ctx.check("reorder_steps", labeled(4, 1, 5), np.zeros((4, 1), dtype=np.int64), **EXACT)
    ctx.check("reorder_steps", labeled(4, 3, 1), perms(ctx, 4, 3), **EXACT)
    ctx.check("reorder_steps", labeled(1, 1, 1), np.zeros((1, 1), dtype=np.int64), **EXACT)
