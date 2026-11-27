"""Tests for sort-and-unsort.

Every case names the function it tests. Indices and masks are compared exactly. Index
tensors are passed as numpy int64 and arrive as the framework's default integer type:
int64 in PyTorch, int32 in JAX and MLX. Rows with equal values are everywhere, because
the tie rule (lowest index first) is part of the specification.
"""

import numpy as np

from learnai.judge import case, check


def floats(values):
    return np.array(values, dtype=np.float32)


def ints(values):
    return np.array(values, dtype=np.int64)


def bools(rows):
    """A boolean array from rows written as strings: "#" is true, "." is false."""
    return np.array([[c == "#" for c in row] for row in rows], dtype=bool)


def perms(ctx, batch, size):
    """One random permutation of 0 .. size - 1 per row."""
    return np.stack([ctx.rng.permutation(size) for _ in range(batch)]).astype(np.int64)


def with_ties(ctx, batch, size, distinct):
    """Rows drawn from `distinct` different values, so most values occur several times."""
    return ctx.rng.integers(0, distinct, size=(batch, size)).astype(np.float32)


def take(values, index):
    return np.take_along_axis(values, index.astype(np.int64), axis=1)


def check_rank(ctx, x):
    """Compare with the oracle, then state the meaning in numpy."""
    order = ctx.check("rank_descending", x).astype(np.int64)
    ranked = take(x, order)
    check.that(
        bool((ranked[:, :-1] >= ranked[:, 1:]).all()),
        "rank_descending: the values read in the returned order are not largest first",
        f"x\n{x}\n\nreturned order\n{order}\n\nx in that order\n{ranked}",
    )
    return order


# ------------------------------------------------------------------- rank_descending


# Rejects rank_ties_highest_index (row 0 would start 3, 1, 2, 0), rank_returns_positions
# (neither row's order is its own inverse), and rank_sorts_batch_axis.
@case("rank_descending: the worked example, x of shape (2, 4) with ties", public=True)
def rank_example(ctx):
    x = floats([[0.5, 2.0, 0.5, 3.0], [1.0, 1.0, 4.0, 1.0]])
    got = check_rank(ctx, x)
    check.equal(got, ints([[3, 1, 0, 2], [2, 0, 1, 3]]), label="rank_descending")


@case("rank_descending: a row of equal values keeps its index order, 0 to V - 1", public=True)
def rank_all_equal(ctx):
    x = floats([[1.0, 1.0, 1.0, 1.0, 1.0], [0.0, 2.0, 0.0, 2.0, 0.0]])
    got = check_rank(ctx, x)
    check.equal(got, ints([[0, 1, 2, 3, 4], [1, 3, 0, 2, 4]]), label="rank_descending")


# Rejects rank_bare_squeeze: the result must be (1, 4), and squeeze() returns (4,).
@case("rank_descending: a batch of one returns shape (1, 4)", public=True)
def rank_batch_of_one(ctx):
    check_rank(ctx, floats([[0.5, 2.0, 0.5, 3.0]]))


# Rejects rank_unstable_sort: on a row this long torch's default sort does not keep
# equal values in index order. Row 0 alternates 1, 0, 1, 0, ...; row 1 has four values.
@case(
    "rank_descending: 64 values of which only a few are different, x of shape (2, 64)", public=True
)
def rank_tie_heavy(ctx):
    x = np.stack([np.tile(floats([1.0, 0.0]), 32), np.tile(floats([2.0, 0.0, 3.0, 1.0]), 16)])
    got = check_rank(ctx, x)
    ones_first = np.concatenate([np.arange(0, 64, 2), np.arange(1, 64, 2)])
    check.equal(got[0], ones_first, label="rank_descending, row 0")


# Rejects rank_epsilon_tiebreak: the values are 100 and 99, where float32 cannot hold a
# bonus of 1e-6 per column, so the bonus is rounded away or merges neighboring columns.
@case("rank_descending: 64 values that are all 100 or 99, x of shape (2, 64)", public=True)
def rank_ties_at_100(ctx):
    x = np.stack(
        [np.tile(floats([99.0, 100.0]), 32), np.tile(floats([100.0, 100.0, 99.0]), 22)[:64]]
    )
    got = check_rank(ctx, x)
    hundreds_first = np.concatenate([np.arange(1, 64, 2), np.arange(0, 64, 2)])
    check.equal(got[0], hundreds_first, label="rank_descending, row 0")


# Rejects rank_epsilon_tiebreak: a larger value in a late column must still come
# first, however small the difference; a per-column bonus outweighs it.
@case(
    "rank_descending: values 1e-5 apart are not ties, the larger one in the last column",
    public=True,
)
def rank_close_values(ctx):
    x = np.ones((2, 64), dtype=np.float32)
    x[0, 63] = 1.00001
    x[1, 40] = 1.00001
    x[1, 5] = 0.99999
    got = check_rank(ctx, x)
    check.that(
        int(got[0, 0]) == 63,
        f"rank_descending: row 0 ranks column {int(got[0, 0])} first; column 63 holds the "
        "largest value, 1.00001 against 1.0 everywhere else",
        f"returned order, row 0\n{got[0]}",
    )


@case("rank_descending: as many rows as columns, B = V = 4, with and without ties")
def rank_square(ctx):
    check_rank(ctx, floats([[3, 1, 2, 0], [0, 0, 5, 5], [7, 7, 7, 7], [1, 4, 1, 4]]))
    check_rank(ctx, ctx.randn(4, 4))


@case("rank_descending: random rows, distinct values and values with ties, B = 4, V = 8")
def rank_random(ctx):
    for _ in range(2):
        check_rank(ctx, ctx.randn(4, 8))
        check_rank(ctx, with_ties(ctx, 4, 8, 3) - 1.0)


@case("rank_descending: a single column, and a (1, 1) input")
def rank_size_one(ctx):
    check_rank(ctx, floats([[3.0], [1.0], [3.0], [2.0]]))
    check_rank(ctx, floats([[5.0]]))


# --------------------------------------------------------------- inverse_permutation


def check_inverse(ctx, perm):
    """Compare with the oracle, then state the meaning: out[b, perm[b, k]] = k."""
    inv = ctx.check("inverse_permutation", perm).astype(np.int64)
    ranks = np.broadcast_to(np.arange(perm.shape[1]), perm.shape)
    check.equal(take(inv, perm), ranks, label="the result read at perm (must be 0 .. V - 1)")
    return inv


# Rejects inverse_by_gather (it returns perm) and inverse_on_batch_axis.
@case("inverse_permutation: the worked example, perm of shape (2, 4)", public=True)
def inverse_example(ctx):
    got = check_inverse(ctx, ints([[3, 1, 0, 2], [2, 0, 1, 3]]))
    check.equal(got, ints([[2, 1, 3, 0], [1, 2, 0, 3]]), label="inverse_permutation")


@case("inverse_permutation: a batch of one, perm [[1, 2, 3, 0]]", public=True)
def inverse_batch_of_one(ctx):
    got = check_inverse(ctx, ints([[1, 2, 3, 0]]))
    check.equal(got, ints([[3, 0, 1, 2]]), label="inverse_permutation")


# inverse_by_gather passes this case, as its BUG line says.
@case("inverse_permutation: the identity and a swap of neighbors are their own inverses")
def inverse_involutions(ctx):
    perm = ints([[0, 1, 2, 3], [1, 0, 3, 2]])
    got = check_inverse(ctx, perm)
    check.equal(got, perm, label="inverse_permutation")


@case("inverse_permutation: as many rows as columns, B = V = 4, and random rows of 8")
def inverse_random(ctx):
    check_inverse(ctx, ints([[1, 2, 3, 0], [3, 0, 1, 2], [2, 3, 1, 0], [0, 2, 3, 1]]))
    for _ in range(3):
        check_inverse(ctx, perms(ctx, 4, 8))


@case("inverse_permutation: a single column, and a (1, 1) input")
def inverse_size_one(ctx):
    check_inverse(ctx, np.zeros((4, 1), dtype=np.int64))
    check_inverse(ctx, np.zeros((1, 1), dtype=np.int64))


# ------------------------------------------------------------ keep_to_original_order


# Rejects keep_gathers_with_order (row 0 would be . # # .), keep_first_row_order, and
# keep_as_float (dtype float32).
@case(
    "keep_to_original_order: the worked example, the first two ranks of row 0 and the first of row 1",
    public=True,
)
def keep_example(ctx):
    order = ints([[3, 1, 0, 2], [2, 0, 1, 3]])
    got = ctx.check("keep_to_original_order", bools(["##..", "#..."]), order)
    check.equal(got, bools([".#.#", "..#."]), label="keep_to_original_order")


@case("keep_to_original_order: a batch of one, order [[1, 2, 3, 0]]", public=True)
def keep_batch_of_one(ctx):
    got = ctx.check("keep_to_original_order", bools(["#.#."]), ints([[1, 2, 3, 0]]))
    check.equal(got, bools([".#.#"]), label="keep_to_original_order")


# The whole pattern. The order is the reference's own rank_descending, so this case
# grades keep_to_original_order alone.
@case("keep_to_original_order: the two largest of every row, decided in sorted order", public=True)
def keep_top_two(ctx):
    x = floats([[0.5, 2.0, 0.5, 3.0, 1.0], [1.0, 1.0, 4.0, 1.0, 0.0], [5.0, 5.0, 5.0, 5.0, 5.0]])
    order = ctx.expected("rank_descending", x).astype(np.int64)
    keep_sorted = np.broadcast_to(np.arange(5) < 2, (3, 5)).copy()
    got = ctx.check("keep_to_original_order", keep_sorted, order)
    check.equal(got, bools([".#.#.", "#.#..", "##..."]), label="keep_to_original_order")


@case("keep_to_original_order: every rank kept, and no rank kept")
def keep_all_and_none(ctx):
    order = perms(ctx, 4, 8)
    for value in (True, False):
        ctx.check("keep_to_original_order", np.full((4, 8), value), order)


@case("keep_to_original_order: as many rows as columns, B = V = 4, and random rows of 8")
def keep_random(ctx):
    order = ints([[1, 2, 3, 0], [3, 0, 1, 2], [2, 3, 1, 0], [0, 2, 3, 1]])
    ctx.check("keep_to_original_order", bools(["#...", "##..", ".#.#", "###."]), order)
    for _ in range(3):
        ctx.check("keep_to_original_order", ctx.rng.random((4, 8)) < 0.5, perms(ctx, 4, 8))


@case("keep_to_original_order: a single column, and a (1, 1) input")
def keep_size_one(ctx):
    zeros = np.zeros((4, 1), dtype=np.int64)
    ctx.check("keep_to_original_order", bools(["#", ".", ".", "#"]), zeros)
    ctx.check("keep_to_original_order", bools(["#"]), zeros[:1])


# ------------------------------------------------------------------ all three together


@case("all three: sorting and then unsorting returns every row to where it was, B = 4, V = 8")
def round_trip(ctx):
    def prop(rng):
        x = rng.integers(-2, 3, size=(4, 8)).astype(np.float32)
        order = ctx.call("rank_descending", x).astype(np.int64)
        inv = ctx.call("inverse_permutation", order).astype(np.int64)
        detail = f"x\n{x}\n\nrank_descending(x)\n{order}\n\ninverse_permutation of that\n{inv}"
        check.that(
            np.array_equal(take(take(x, order), inv), x),
            "sorting x with rank_descending and reading the result through "
            "inverse_permutation does not give x back",
            detail,
        )
        back = ctx.call("inverse_permutation", inv)
        check.equal(back, order, label="inverse_permutation applied twice")
        keep = rng.random((4, 8)) < 0.5
        got = ctx.call("keep_to_original_order", take(keep, order), order)
        check.equal(got, keep, label="a mask sorted with the order and moved back")

    check.forall(prop, trials=5)
