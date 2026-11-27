"""Tests for paired-and-outer-indexing.

Every case names the function it tests. Indexing copies values, so every comparison is
exact. Index tensors are passed as numpy int64 and arrive as the framework's default
integer type.
"""

import numpy as np

from learnai.judge import case, check

EXACT = {"rtol": 0.0, "atol": 0.0}


def idx(*values):
    return np.array(values, dtype=np.int64)


def table(height, width):
    """x[i, j] = 10 i + j: every entry names its own row and column."""
    return (10 * np.arange(height)[:, None] + np.arange(width)[None, :]).astype(np.float32)


def labeled(batch, seq, width):
    """x[b, t, d] = 100 b + 10 t + d: every entry names its own position."""
    b, t, d = np.meshgrid(np.arange(batch), np.arange(seq), np.arange(width), indexing="ij")
    return (100 * b + 10 * t + d).astype(np.float32)


# ------------------------------------------------------------------------ pick_pairs


# Rejects pairs_by_chained_indexing (shape (3, 3)) and pairs_swapped (wrong values).
@case("pick_pairs: the worked example, 3 pairs from a (3, 4) table", public=True)
def pairs_example(ctx):
    got = ctx.check("pick_pairs", table(3, 4), idx(2, 0, 1), idx(1, 2, 0), **EXACT)
    check.allclose(got, np.array([21.0, 2.0, 10.0], dtype=np.float32), label="pick_pairs", **EXACT)


@case("pick_pairs: more pairs than rows, with a pair that repeats", public=True)
def pairs_repeats(ctx):
    ctx.check("pick_pairs", table(3, 4), idx(2, 0, 2, 1, 2), idx(3, 3, 0, 1, 3), **EXACT)


# With N = H = W every wrong pairing has an index in range and some shape of size 4.
@case("pick_pairs: as many pairs as rows and columns, N = H = W = 4")
def pairs_square(ctx):
    ctx.check("pick_pairs", table(4, 4), idx(3, 0, 2, 1), idx(1, 2, 0, 0), **EXACT)
    ctx.check("pick_pairs", ctx.randn(4, 4), idx(0, 1, 3, 3), idx(2, 3, 0, 1), **EXACT)


@case("pick_pairs: a single pair, a table of one row, and a table of one entry")
def pairs_size_one(ctx):
    ctx.check("pick_pairs", table(3, 4), idx(1), idx(3), **EXACT)
    ctx.check("pick_pairs", table(1, 4), idx(0, 0, 0), idx(3, 0, 2), **EXACT)
    ctx.check("pick_pairs", table(1, 1), idx(0), idx(0), **EXACT)


# ------------------------------------------------------------------------ pick_block


@case("pick_block: the worked example, 2 rows by 3 columns of a (3, 4) table", public=True)
def block_example(ctx):
    got = ctx.check("pick_block", table(3, 4), idx(2, 0), idx(3, 1, 0), **EXACT)
    expected = np.array([[23.0, 21.0, 20.0], [3.0, 1.0, 0.0]], dtype=np.float32)
    check.allclose(got, expected, label="pick_block", **EXACT)


# Rejects block_by_paired_indexing (it returns the 3 paired entries, shape (3,)) and
# block_transposed (wrong values).
@case("pick_block: as many rows as columns, R = C = 3", public=True)
def block_square(ctx):
    ctx.check("pick_block", table(3, 4), idx(2, 0, 1), idx(3, 1, 0), **EXACT)


@case("pick_block: R = C = H = W = 4, and indices that repeat")
def block_all_equal(ctx):
    ctx.check("pick_block", table(4, 4), idx(3, 0, 2, 1), idx(1, 2, 0, 3), **EXACT)
    ctx.check("pick_block", ctx.randn(4, 4), idx(2, 2, 0, 2), idx(1, 3, 3, 1), **EXACT)


@case("pick_block: more chosen rows than the table has, R = 5 and C = 2")
def block_tall(ctx):
    ctx.check("pick_block", table(3, 4), idx(2, 0, 2, 1, 1), idx(3, 0), **EXACT)


@case("pick_block: a single row, a single column, and a single entry")
def block_size_one(ctx):
    ctx.check("pick_block", table(3, 4), idx(1), idx(3, 1, 0), **EXACT)
    ctx.check("pick_block", table(3, 4), idx(2, 0, 1), idx(2), **EXACT)
    ctx.check("pick_block", table(1, 1), idx(0), idx(0), **EXACT)


# ---------------------------------------------------------------------- pick_per_row


# Rejects per_row_by_slice (shape (2, 2, 3, 2)) and per_row_first_row_only.
@case("pick_per_row: the worked example, 3 steps from each of 2 rows", public=True)
def per_row_example(ctx):
    cols = np.array([[2, 0, 2], [1, 1, 0]], dtype=np.int64)
    got = ctx.check("pick_per_row", labeled(2, 3, 2), cols, **EXACT)
    expected = np.array(
        [
            [[20.0, 21.0], [0.0, 1.0], [20.0, 21.0]],
            [[110.0, 111.0], [110.0, 111.0], [100.0, 101.0]],
        ],
        dtype=np.float32,
    )
    check.allclose(got, expected, label="pick_per_row", **EXACT)


# Rejects per_row_batch_not_a_column with wrong values: (B,) lines up with the K axis.
@case("pick_per_row: as many picks as rows, B = K = 3 with T = 5 and D = 2", public=True)
def per_row_batch_equals_picks(ctx):
    cols = np.array([[4, 0, 2], [1, 1, 3], [0, 4, 4]], dtype=np.int64)
    ctx.check("pick_per_row", labeled(3, 5, 2), cols, **EXACT)


@case("pick_per_row: all sizes differ, B = 2, T = 5, D = 3, K = 4")
def per_row_distinct(ctx):
    ctx.check("pick_per_row", ctx.randn(2, 5, 3), ctx.randint(0, 5, (2, 4)), **EXACT)


@case("pick_per_row: all sizes equal, B = T = D = K = 3")
def per_row_all_equal(ctx):
    cols = np.array([[2, 0, 1], [1, 1, 0], [0, 2, 2]], dtype=np.int64)
    ctx.check("pick_per_row", labeled(3, 3, 3), cols, **EXACT)
    ctx.check("pick_per_row", ctx.randn(3, 3, 3), ctx.randint(0, 3, (3, 3)), **EXACT)


@case("pick_per_row: one pick per row, a batch of one, one feature, and all sizes 1")
def per_row_size_one(ctx):
    ctx.check("pick_per_row", labeled(3, 5, 2), np.array([[4], [0], [2]], dtype=np.int64), **EXACT)
    ctx.check("pick_per_row", labeled(1, 5, 2), np.array([[3, 0, 3, 1]], dtype=np.int64), **EXACT)
    ctx.check("pick_per_row", labeled(3, 5, 1), ctx.randint(0, 5, (3, 4)), **EXACT)
    ctx.check("pick_per_row", labeled(1, 1, 1), np.zeros((1, 1), dtype=np.int64), **EXACT)
