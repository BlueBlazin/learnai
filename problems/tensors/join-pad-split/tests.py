"""Tests for join-pad-split.

Every case names the function it tests. All results are copies of input values (or the
pad and bos ids), so every comparison is exact. Token ids are passed as numpy int64 and
arrive as the framework's default integer type; a list of arrays arrives as a list of
tensors.
"""

import numpy as np

from learnai.judge import case, check

EXACT = {"rtol": 0.0, "atol": 0.0}


def ints(values):
    return np.array(values, dtype=np.int64)


def rows_of(*rows):
    return [ints(row) for row in rows]


def bools(rows):
    """A boolean array from rows written as strings: "#" is true, "." is false."""
    return np.array([[c == "#" for c in row] for row in rows], dtype=bool)


def check_pad(ctx, rows, pad, side, ids=None, mask=None):
    got_ids, got_mask = ctx.check("pad_rows", rows, pad, side)
    if ids is not None:
        check.equal(got_ids, ints(ids), label="pad_rows[0]")
        check.equal(got_mask, bools(mask), label="pad_rows[1]")


def check_shift(ctx, ids, bos):
    """Compare with the reference, after naming a wrapped token if there is one."""
    got = np.asarray(ctx.call("shift_right", ids, bos))
    if ids.shape[1] == 1 and np.array_equal(got, ids) and (ids != bos).any():
        b = int(np.argmax(ids[:, 0] != bos))
        check.that(
            False,
            f"shift_right: a single position came back unchanged ({int(got[b, 0])} in row {b}), "
            f"expected bos = {bos}: with T = 1 the only token is dropped and bos is all that remains",
            f"ids = {ids.tolist()}\ngot = {got.tolist()}",
        )
    if got.shape == ids.shape and np.array_equal(got, np.roll(ids, 1, axis=1)):
        wrapped = ids[:, -1] != bos
        if wrapped.any():
            b = int(np.argmax(wrapped))
            check.that(
                False,
                f"shift_right: row {b} starts with {int(got[b, 0])}, the last token of that "
                f"row, expected bos = {bos}: the last column wrapped around to the front",
                f"ids[{b}] = {ids[b].tolist()}\ngot[{b}] = {got[b].tolist()}",
            )
    return ctx.check("shift_right", ids, bos)


def step_list(batch, steps, vocab):
    """steps[t][b, v] = 100 t + 10 b + v: every entry names its step, row, and column."""
    b, v = np.meshgrid(np.arange(batch), np.arange(vocab), indexing="ij")
    return [(100 * t + 10 * b + v).astype(np.float32) for t in range(steps)]


# -------------------------------------------------------------------------- pad_rows

EXAMPLE = ([5, 6, 7], [8], [9, 10])


# Rejects pad_sides_swapped, pad_mask_inverted, and pad_mask_as_integers.
@case('pad_rows: the worked example, rows of 3, 1, and 2 tokens, side="left"', public=True)
def pad_example_left(ctx):
    check_pad(
        ctx,
        rows_of(*EXAMPLE),
        0,
        "left",
        ids=[[5, 6, 7], [0, 0, 8], [0, 9, 10]],
        mask=["###", "..#", ".##"],
    )


@case('pad_rows: the same rows, side="right"', public=True)
def pad_example_right(ctx):
    check_pad(
        ctx,
        rows_of(*EXAMPLE),
        0,
        "right",
        ids=[[5, 6, 7], [8, 0, 0], [9, 10, 0]],
        mask=["###", "#..", "##."],
    )


# Rejects pad_mask_from_values: the 2s inside the rows are real tokens.
# The longest row is in the middle: a width taken from rows[0] is too short.
@case("pad_rows: the pad id also occurs as a real token, pad = 2, longest row second", public=True)
def pad_id_is_a_token(ctx):
    check_pad(
        ctx,
        rows_of([2], [7, 2, 9, 2], [4, 2]),
        2,
        "left",
        ids=[[2, 2, 2, 2], [7, 2, 9, 2], [2, 2, 4, 2]],
        mask=["...#", "####", "..##"],
    )
    check_pad(ctx, rows_of([2], [7, 2, 9, 2], [4, 2]), 2, "right")


@case("pad_rows: a batch of one row returns shape (1, 4) and an all-true mask", public=True)
def pad_batch_of_one(ctx):
    for side in ("left", "right"):
        check_pad(ctx, rows_of([3, 1, 4, 1]), 0, side, ids=[[3, 1, 4, 1]], mask=["####"])


@case("pad_rows: rows of equal length need no padding, B = T_max = 3")
def pad_equal_lengths(ctx):
    rows = rows_of([1, 2, 3], [4, 5, 6], [7, 8, 9])
    for side in ("left", "right"):
        check_pad(ctx, rows, 0, side, ids=[[1, 2, 3], [4, 5, 6], [7, 8, 9]], mask=["###"] * 3)


@case("pad_rows: every row has one token, and a single row of one token")
def pad_single_tokens(ctx):
    for side in ("left", "right"):
        check_pad(ctx, rows_of([5], [6], [7]), 9, side, ids=[[5], [6], [7]], mask=["#"] * 3)
        check_pad(ctx, rows_of([5]), 9, side, ids=[[5]], mask=["#"])


@case("pad_rows: random rows of 1 to 4 tokens, B = 6, a nonzero pad id, both sides")
def pad_random(ctx):
    for _ in range(3):
        lengths = [4, *ctx.rng.integers(1, 5, size=5).tolist()]
        rows = [ctx.randint(0, 1000, (n,)) for n in lengths]
        for side in ("left", "right"):
            check_pad(ctx, rows, 999, side)


@case("pad_rows: the longest row is last, and is 12 tokens long")
def pad_longest_last(ctx):
    rows = [ctx.randint(0, 1000, (n,)) for n in (1, 3, 12)]
    for side in ("left", "right"):
        check_pad(ctx, rows, 0, side)


@case("pad_rows: the largest batch, 8 rows of 1 to 12 tokens")
def pad_largest(ctx):
    rows = [ctx.randint(0, 1000, (n,)) for n in (12, 1, 4, 4, 3, 12, 1, 3)]
    for side in ("left", "right"):
        check_pad(ctx, rows, 0, side)


# ----------------------------------------------------------------------- shift_right


# Rejects shift_by_roll (4 and 8 in column 0), shift_keeps_last_column (shape (2, 5)),
# and shift_left.
@case("shift_right: the worked example, ids of shape (2, 4), bos = 99", public=True)
def shift_example(ctx):
    got = check_shift(ctx, ints([[1, 2, 3, 4], [5, 6, 7, 8]]), 99)
    check.equal(got, ints([[99, 1, 2, 3], [99, 5, 6, 7]]), label="shift_right")


@case("shift_right: a batch of one, ids of shape (1, 5), bos = 0", public=True)
def shift_batch_of_one(ctx):
    got = check_shift(ctx, ints([[3, 1, 4, 1, 5]]), 0)
    check.equal(got, ints([[0, 3, 1, 4, 1]]), label="shift_right")


@case("shift_right: a single position holds bos alone, ids of shape (3, 1)", public=True)
def shift_one_position(ctx):
    got = check_shift(ctx, ints([[4], [5], [6]]), 99)
    check.equal(got, ints([[99], [99], [99]]), label="shift_right")


@case("shift_right: as many rows as positions, B = T = 4, and random ids of shape (5, 7)")
def shift_square(ctx):
    check_shift(ctx, ints([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]), 0)
    for _ in range(2):
        check_shift(ctx, ctx.randint(0, 1000, (5, 7)), 1)


@case("shift_right: the largest sizes, B = T = 8")
def shift_largest(ctx):
    check_shift(ctx, ctx.randint(0, 1000, (8, 8)), 999)


@case("shift_right: bos already occurs in the rows, and a (1, 1) input")
def shift_bos_in_rows(ctx):
    got = check_shift(ctx, ints([[1, 7, 2, 7], [1, 1, 1, 1]]), 1)
    check.equal(got, ints([[1, 1, 7, 2], [1, 1, 1, 1]]), label="shift_right")
    check_shift(ctx, ints([[4]]), 99)


# ----------------------------------------------------------------------- stack_steps


# Rejects stack_on_axis_0 (shape (3, 2, 2)) and stack_by_cat (shape (2, 6)).
@case("stack_steps: the worked example, 3 steps of shape (2, 2)", public=True)
def stack_example(ctx):
    got = ctx.check("stack_steps", step_list(2, 3, 2), **EXACT)
    expected = np.array(
        [
            [[0, 1], [100, 101], [200, 201]],
            [[10, 11], [110, 111], [210, 211]],
        ],
        dtype=np.float32,
    )
    check.allclose(got, expected, label="stack_steps", **EXACT)


# Rejects stack_bare_squeeze: the result must be (1, 3, 4), and squeeze() returns (3, 4).
@case("stack_steps: a batch of one keeps its batch axis, 3 steps of shape (1, 4)", public=True)
def stack_batch_of_one(ctx):
    ctx.check("stack_steps", step_list(1, 3, 4), **EXACT)


# stack_on_axis_0 returns the right shape here, with rows and steps exchanged.
@case("stack_steps: as many steps as rows, B = T = 3, V = 4", public=True)
def stack_steps_equal_rows(ctx):
    got = ctx.check("stack_steps", step_list(3, 3, 4), **EXACT)
    check.allclose(got[0, 2], np.array([200, 201, 202, 203]), label="stack_steps[0, 2]", **EXACT)


# stack_on_last_axis returns the right shape here.
@case("stack_steps: as many steps as columns, T = V = 4, and all sizes equal")
def stack_steps_equal_columns(ctx):
    ctx.check("stack_steps", step_list(2, 4, 4), **EXACT)
    ctx.check("stack_steps", step_list(3, 3, 3), **EXACT)


@case("stack_steps: random values, 5 steps of shape (2, 3)")
def stack_random(ctx):
    for _ in range(2):
        ctx.check("stack_steps", [ctx.randn(2, 3) for _ in range(5)], **EXACT)


@case("stack_steps: the largest sizes, 8 steps of shape (8, 8)")
def stack_largest(ctx):
    ctx.check("stack_steps", [ctx.randn(8, 8) for _ in range(8)], **EXACT)


@case("stack_steps: a single step returns (B, 1, V); one column; and all sizes 1")
def stack_size_one(ctx):
    ctx.check("stack_steps", step_list(2, 1, 4), **EXACT)
    ctx.check("stack_steps", step_list(2, 3, 1), **EXACT)
    ctx.check("stack_steps", step_list(1, 1, 1), **EXACT)
