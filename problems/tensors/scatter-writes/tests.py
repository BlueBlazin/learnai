"""Tests for scatter-writes.

Every case names the function it tests. Writes copy values and counts are integers, so
every comparison is exact. Index tensors are passed as numpy int64 and arrive as the
framework's default integer type: int64 in PyTorch, int32 in JAX and MLX.
"""

import numpy as np

from learnai.judge import case, check

EXACT = {"rtol": 0.0, "atol": 0.0}


def ints(values):
    return np.array(values, dtype=np.int64)


def table(rows, cols):
    """x[b, v] = 10 b + v: every entry names its own row and column."""
    return (10 * np.arange(rows)[:, None] + np.arange(cols)[None, :]).astype(np.float32)


def new_values(rows, picks):
    """values[b, k] = 900 + 10 b + k."""
    return (900 + 10 * np.arange(rows)[:, None] + np.arange(picks)[None, :]).astype(np.float32)


def distinct_columns(ctx, rows, cols, picks):
    """picks distinct columns per row, in random order."""
    return np.stack([ctx.rng.permutation(cols)[:picks] for _ in range(rows)]).astype(np.int64)


# ---------------------------------------------------------------------- one_hot_rows


# Rejects uses_builtin (the call is refused) and one_hot_integer (dtype int64).
@case("one_hot_rows: the worked example, ids [2, 0, 2] with V = 4", public=True)
def one_hot_example(ctx):
    got = ctx.check("one_hot_rows", ints([2, 0, 2]), 4, **EXACT)
    expected = np.array([[0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 1, 0]], dtype=np.float32)
    check.allclose(got, expected, label="one_hot_rows", **EXACT)


@case("one_hot_rows: a batch of one returns shape (1, 5)", public=True)
def one_hot_batch_of_one(ctx):
    ctx.check("one_hot_rows", ints([3]), 5, **EXACT)


# Rejects one_hot_scatter_on_batch_axis with wrong values: every id is a valid row
# number, and the result is the transpose of the expected table.
@case("one_hot_rows: as many rows as columns, B = V = 3, ids [1, 2, 2]", public=True)
def one_hot_square(ctx):
    ctx.check("one_hot_rows", ints([1, 2, 2]), 3, **EXACT)


@case("one_hot_rows: ids drawn at random, B = 5, V = 7")
def one_hot_random(ctx):
    for _ in range(3):
        ctx.check("one_hot_rows", ctx.randint(0, 7, (5,)), 7, **EXACT)


@case("one_hot_rows: a single column, and a (1, 1) result")
def one_hot_size_one(ctx):
    ctx.check("one_hot_rows", np.zeros(4, dtype=np.int64), 1, **EXACT)
    ctx.check("one_hot_rows", np.zeros(1, dtype=np.int64), 1, **EXACT)


# ---------------------------------------------------------------------- token_counts


# Rejects counts_by_setting_scatter and counts_by_indexed_increment (token 1 of row 0
# and token 2 of row 1 are counted once), counts_over_whole_batch, and counts_as_float.
@case("token_counts: the worked example, ids of shape (2, 3) with V = 4", public=True)
def counts_example(ctx):
    got = ctx.check("token_counts", ints([[1, 1, 3], [0, 2, 2]]), 4, **EXACT)
    check.equal(got, np.array([[0, 2, 0, 1], [1, 0, 2, 0]]), label="token_counts")


# counts_over_whole_batch passes this case, as its BUG line says.
@case("token_counts: a batch of one with a token three times, ids [[2, 2, 2, 0]]", public=True)
def counts_batch_of_one(ctx):
    got = ctx.check("token_counts", ints([[2, 2, 2, 0]]), 3, **EXACT)
    check.equal(got, np.array([[1, 0, 3]]), label="token_counts")


@case("token_counts: every row of the result sums to T, B = 4, T = 6, V = 5", public=True)
def counts_row_sums(ctx):
    ids = ctx.randint(0, 5, (4, 6))
    got = ctx.check("token_counts", ids, 5, **EXACT)
    check.equal(got.sum(axis=1), np.full(4, 6), label="row sums of token_counts")


@case("token_counts: all sizes equal, B = T = V = 3")
def counts_equal_axes(ctx):
    ctx.check("token_counts", ints([[0, 0, 1], [2, 1, 2], [1, 1, 1]]), 3, **EXACT)
    ctx.check("token_counts", ctx.randint(0, 3, (3, 3)), 3, **EXACT)


@case("token_counts: a row that is one token throughout counts T for it")
def counts_one_token(ctx):
    ids = np.tile(ints([[4], [0], [2], [2]]), (1, 6))
    ctx.check("token_counts", ids, 5, **EXACT)


@case("token_counts: one position per row, a vocabulary of one token, and all sizes 1")
def counts_size_one(ctx):
    ctx.check("token_counts", ints([[3], [0], [3], [1]]), 5, **EXACT)
    ctx.check("token_counts", np.zeros((4, 6), dtype=np.int64), 1, **EXACT)
    ctx.check("token_counts", np.zeros((1, 1), dtype=np.int64), 1, **EXACT)


# -------------------------------------------------------------------------- write_at


# Rejects write_on_batch_axis (index 3 is not a row: it raises), write_index_as_value
# (row 1 gets 1 and 2 where 910 and 911 belong), and write_first_row_index.
@case("write_at: the worked example, 2 writes into each row of a (2, 4) table", public=True)
def write_example(ctx):
    idx = ints([[3, 0], [1, 2]])
    got = ctx.check("write_at", table(2, 4), idx, new_values(2, 2), **EXACT)
    expected = np.array([[901.0, 1.0, 2.0, 900.0], [10.0, 910.0, 911.0, 13.0]], dtype=np.float32)
    check.allclose(got, expected, label="write_at", **EXACT)


@case("write_at: a batch of one, x of shape (1, 4)", public=True)
def write_batch_of_one(ctx):
    ctx.check("write_at", table(1, 4), ints([[2, 0]]), new_values(1, 2), **EXACT)


# Rejects write_on_batch_axis with wrong values: every index is a valid row number.
@case("write_at: more rows than columns, B = 5, V = 3, K = 2", public=True)
def write_tall(ctx):
    idx = ints([[2, 0], [1, 2], [0, 1], [2, 1], [0, 2]])
    ctx.check("write_at", table(5, 3), idx, new_values(5, 2), **EXACT)


# Rejects write_in_place. The solution is called directly, on tensors this test
# holds, because ctx.check hands every call a fresh copy of its inputs.
@case("write_at: x itself is unchanged after the call", public=True)
def write_leaves_x(ctx):
    before = table(2, 4)
    idx = ints([[3, 0], [1, 2]])
    values = new_values(2, 2)
    x = ctx.fw.tensor(before)
    out = ctx.fw.to_numpy(ctx.solution.write_at(x, ctx.fw.tensor(idx), ctx.fw.tensor(values)))
    after = ctx.fw.to_numpy(x)
    check.allclose(out, ctx.expected("write_at", before, idx, values), label="write_at", **EXACT)
    changed = int((after != before).sum())
    check.that(
        changed == 0,
        f"write_at: the call changed {changed} of 8 entries of its argument x; "
        "x must be left as it was and the result returned as a new tensor",
        f"x before the call\n{before}\n\nx after the call\n{after}",
    )


@case("write_at: all sizes equal, B = V = K = 3, every entry of x replaced")
def write_all_equal(ctx):
    idx = ints([[2, 0, 1], [1, 2, 0], [0, 2, 1]])
    ctx.check("write_at", table(3, 3), idx, new_values(3, 3), **EXACT)
    ctx.check("write_at", ctx.randn(3, 3), distinct_columns(ctx, 3, 3, 3), ctx.randn(3, 3), **EXACT)


@case("write_at: random distinct columns, B = 4, V = 6, K = 3")
def write_random(ctx):
    for _ in range(3):
        idx = distinct_columns(ctx, 4, 6, 3)
        ctx.check("write_at", ctx.randn(4, 6), idx, ctx.randn(4, 3), **EXACT)


@case("write_at: one write per row, a single column, and a (1, 1) table")
def write_size_one(ctx):
    ctx.check("write_at", table(5, 3), ints([[2], [0], [1], [1], [0]]), new_values(5, 1), **EXACT)
    ctx.check("write_at", table(4, 1), np.zeros((4, 1), dtype=np.int64), new_values(4, 1), **EXACT)
    ctx.check("write_at", table(1, 1), np.zeros((1, 1), dtype=np.int64), new_values(1, 1), **EXACT)
