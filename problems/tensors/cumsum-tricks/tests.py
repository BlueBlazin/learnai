"""Tests for cumsum-tricks.

Every case names the function it tests. first_true and through_first_eos return
integers and booleans and are compared exactly. exclusive_cumsum is a float sum:
inputs of whole numbers are exact in float32, and random inputs are compared with an
absolute tolerance that also accepts the sum computed as cumsum(x) - x.
"""

import numpy as np

from learnai.judge import case, check

SUM_TOL = {"rtol": 1e-4, "atol": 1e-4}


def floats(values):
    return np.array(values, dtype=np.float32)


def ints(values):
    return np.array(values, dtype=np.int64)


def bools(rows):
    """A boolean array from rows written as strings: "#" is true, "." is false."""
    return np.array([[c == "#" for c in row] for row in rows], dtype=bool)


# ------------------------------------------------------------------ exclusive_cumsum


# Rejects cumsum_inclusive, cumsum_shift_wraps (10 and 5 in column 0), and
# cumsum_on_batch_axis.
@case("exclusive_cumsum: the worked example, x of shape (2, 4)", public=True)
def cumsum_example(ctx):
    x = floats([[1, 2, 3, 4], [10, 0, -10, 5]])
    got = ctx.check("exclusive_cumsum", x, **SUM_TOL)
    check.allclose(got, floats([[0, 1, 3, 6], [0, 10, 10, 0]]), label="exclusive_cumsum")


@case("exclusive_cumsum: one axis, and three axes of shape (2, 3, 4)", public=True)
def cumsum_ranks(ctx):
    got = ctx.check("exclusive_cumsum", floats([5, 1, 2]), **SUM_TOL)
    check.allclose(got, floats([0, 5, 6]), label="exclusive_cumsum")
    x = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    ctx.check("exclusive_cumsum", x, **SUM_TOL)


@case("exclusive_cumsum: a batch of one, x of shape (1, 4)", public=True)
def cumsum_batch_of_one(ctx):
    ctx.check("exclusive_cumsum", floats([[3, 1, 4, 1]]), **SUM_TOL)


# cumsum_on_batch_axis returns the right shape for every input; only the values differ.
@case("exclusive_cumsum: equal sizes, (3, 3) and (3, 3, 3)")
def cumsum_equal_axes(ctx):
    ctx.check("exclusive_cumsum", floats([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), **SUM_TOL)
    ctx.check("exclusive_cumsum", np.arange(27, dtype=np.float32).reshape(3, 3, 3), **SUM_TOL)


@case("exclusive_cumsum: random values, shapes (4, 6) and (2, 3, 4)")
def cumsum_random(ctx):
    for _ in range(2):
        ctx.check("exclusive_cumsum", ctx.uniform(4, 6, low=-10.0, high=10.0), **SUM_TOL)
        ctx.check("exclusive_cumsum", ctx.uniform(2, 3, 4, low=-10.0, high=10.0), **SUM_TOL)


# The top of the stated range: 16 entries between -10 and 10.
@case("exclusive_cumsum: 16 entries per row between -10 and 10, x of shape (2, 16)")
def cumsum_full_range(ctx):
    for _ in range(3):
        ctx.check("exclusive_cumsum", ctx.uniform(2, 16, low=-10.0, high=10.0), **SUM_TOL)
    x = np.tile(floats([10.0, -10.0, 10.0, 10.0]), (2, 4))
    ctx.check("exclusive_cumsum", x, **SUM_TOL)


@case("exclusive_cumsum: a last axis of size 1 is all zeros, shapes (3, 1), (1, 1), and (1,)")
def cumsum_size_one(ctx):
    for shape in ((3, 1), (1, 1), (1,)):
        got = ctx.check("exclusive_cumsum", ctx.randn(*shape), **SUM_TOL)
        check.allclose(got, np.zeros(shape, dtype=np.float32), label="exclusive_cumsum")


# ------------------------------------------------------------------------ first_true


# Rejects first_true_by_argmax (row 1 has no true entry: 0 for 5) and
# first_true_counts_false (row 0 has false entries after its first true one: 3 for 1).
@case(
    "first_true: the worked example, rows with several, with none, and with one true entry",
    public=True,
)
def first_example(ctx):
    mask = bools([".#.#.", ".....", "##...", "....#"])
    got = ctx.check("first_true", mask)
    check.equal(got, ints([1, 5, 0, 4]), label="first_true")


@case("first_true: no row has a true entry, so every result is T = 4", public=True)
def first_none(ctx):
    got = ctx.check("first_true", np.zeros((3, 4), dtype=bool))
    check.equal(got, ints([4, 4, 4]), label="first_true")


# Rejects first_true_bare_squeeze: the result must be (1,), and squeeze() returns a
# tensor with no axes.
@case("first_true: a batch of one returns shape (1,)", public=True)
def first_batch_of_one(ctx):
    got = ctx.check("first_true", bools(["..#.#"]))
    check.equal(got, ints([2]), label="first_true")


@case("first_true: as many rows as positions, B = T = 4, and every entry true")
def first_square(ctx):
    ctx.check("first_true", bools(["..##", "#..#", "....", "...#"]))
    ctx.check("first_true", np.ones((4, 4), dtype=bool))


@case("first_true: random masks, B = 5, T = 7, sparse and dense")
def first_random(ctx):
    for density in (0.1, 0.3, 0.7):
        ctx.check("first_true", ctx.rng.random((5, 7)) < density)


@case("first_true: a single position, true and false, and a (1, 1) mask")
def first_size_one(ctx):
    got = ctx.check("first_true", bools(["#", ".", "#"]))
    check.equal(got, ints([0, 1, 0]), label="first_true")
    ctx.check("first_true", bools(["."]))
    ctx.check("first_true", bools(["#"]))


# ----------------------------------------------------------------- through_first_eos


# Rejects eos_dropped (row 0 would stop before its eos), eos_not_equal (row 0 would be
# # . # . #), and eos_until_second (row 0 would stay true up to its second eos).
@case(
    "through_first_eos: the worked example, eos = 2, rows with two, with none, and with three",
    public=True,
)
def eos_example(ctx):
    ids = ints([[5, 2, 7, 2, 9], [1, 3, 4, 6, 8], [2, 2, 5, 2, 5]])
    got = ctx.check("through_first_eos", ids, 2)
    check.equal(got, bools(["##...", "#####", "#...."]), label="through_first_eos")


@case(
    "through_first_eos: eos in the last position keeps the whole row, eos first keeps one",
    public=True,
)
def eos_at_the_ends(ctx):
    ids = ints([[4, 5, 6, 9], [9, 5, 6, 4]])
    got = ctx.check("through_first_eos", ids, 9)
    check.equal(got, bools(["####", "#..."]), label="through_first_eos")


@case("through_first_eos: a batch of one, ids of shape (1, 5)", public=True)
def eos_batch_of_one(ctx):
    got = ctx.check("through_first_eos", ints([[3, 3, 0, 3, 0]]), 0)
    check.equal(got, bools(["###.."]), label="through_first_eos")


@case("through_first_eos: as many rows as positions, B = T = 4, and a row that is all eos")
def eos_square(ctx):
    ids = ints([[1, 1, 1, 1], [0, 1, 0, 1], [0, 0, 0, 0], [0, 0, 1, 0]])
    ctx.check("through_first_eos", ids, 1)


@case("through_first_eos: random ids from a vocabulary of 4, B = 5, T = 7, every eos value")
def eos_random(ctx):
    ids = ctx.randint(0, 4, (5, 7))
    for eos in range(5):  # 4 never occurs: every row is all true
        ctx.check("through_first_eos", ids, eos)


@case("through_first_eos: a single position, eos and not, and a (1, 1) input")
def eos_size_one(ctx):
    got = ctx.check("through_first_eos", ints([[7], [3], [7]]), 7)
    check.equal(got, bools(["#", "#", "#"]), label="through_first_eos")
    ctx.check("through_first_eos", ints([[7]]), 7)
    ctx.check("through_first_eos", ints([[3]]), 7)
