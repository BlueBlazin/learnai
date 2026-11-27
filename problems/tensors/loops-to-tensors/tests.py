"""Tests for loops-to-tensors.

Every case names the function it tests. common_prefix_len and repeated_bigram_ban
return integers and booleans and are compared exactly. segment_means is a float mean:
whole-number inputs with whole-number means are exact, and random inputs are compared
with a tolerance that accepts the sum taken in any order.

The speed cases time the solution against the reference written in the same framework,
with one limit, 20x. In JAX two of them use a quarter of the rows, because a JAX row
loop is the slowest and a learner's slow attempt should still return in seconds.
Sizes: segment_means N = 4096 (JAX 1024), D = 16, S = 8; common_prefix_len B = 2048,
T = 16; repeated_bigram_ban B = 2048 (JAX 512), T = V = 16.

Measured on an M-series laptop, three runs each (slowdown against the reference):

                                PyTorch          JAX             MLX
  segment_means, row loop       more than 75x    more than 500x  3x to 10x (not judged)
  common_prefix_len, row loop   more than 100x   more than 1000x more than 100x
  repeated_bigram_ban, row loop more than 150x   more than 250x  more than 140x
  slowest loop-free solution    12x (ban)        8.2x (prefix)   2x

The loop-free solutions are 30 files from an independent review plus the author's own:
one-hot matrix products, scatter_add with an expanded index, bincount, argmax with a
guard, nonzero followed by a minimum scatter, a (B, T, T) table, a setting scatter
into a spare column, paired boolean indexing with and without unique, a (B, T, V)
comparison. All are accepted,
three runs of three.

In MLX the adding scatter of the reference is itself slow (a one-hot product is 5 to 17
times faster), so a row loop is not 5 times slower than the reference there, and the
segment_means speed case runs for PyTorch and JAX only.

The speed cases do not rule out a loop over a short axis: a loop over the 16 positions
or the 8 segments passes in PyTorch and MLX.
"""

import numpy as np

from learnai.judge import case, check

TOL = {"rtol": 1e-4, "atol": 1e-4}
TIMING = {"rounds": 3, "warmup": 1}


def floats(values):
    return np.array(values, dtype=np.float32)


def ints(values):
    return np.array(values, dtype=np.int64)


def bools(rows):
    """A boolean array from rows written as strings: "#" is true, "." is false."""
    return np.array([[c == "#" for c in row] for row in rows], dtype=bool)


def seg_inputs(ctx, n, d, s):
    return ctx.uniform(n, d, low=-2.0, high=2.0), ctx.randint(0, s, (n,)), s


# --------------------------------------------------------------------- segment_means


# Rejects means_over_all_rows (segment 2 would be [3.75, 37.5]), means_nan_for_empty
# (segments 1 and 3 are empty), and means_scatter_overwrites (segment 2 has three rows).
@case("segment_means: the worked example, 4 rows in 4 segments, two of them empty", public=True)
def means_example(ctx):
    x = floats([[1, 10], [3, 30], [5, 50], [9, 90]])
    got = ctx.check("segment_means", x, ints([2, 0, 2, 2]), 4, **TOL)
    check.allclose(got, floats([[3, 30], [0, 0], [5, 50], [0, 0]]), label="segment_means")


@case("segment_means: all sizes differ, N = 6, D = 3, S = 2, segments not in order", public=True)
def means_distinct(ctx):
    x = floats([[1, 2, 3], [10, 20, 30], [3, 4, 5], [30, 40, 50], [5, 6, 7], [20, 30, 40]])
    got = ctx.check("segment_means", x, ints([1, 0, 1, 0, 1, 0]), 2, **TOL)
    check.allclose(got, floats([[20, 30, 40], [3, 4, 5]]), label="segment_means")


@case("segment_means: one segment returns shape (1, D), and a single row", public=True)
def means_one_segment(ctx):
    x = floats([[1, 10], [3, 30], [5, 50], [7, 70]])
    got = ctx.check("segment_means", x, ints([0, 0, 0, 0]), 1, **TOL)
    check.allclose(got, floats([[4, 40]]), label="segment_means")
    ctx.check("segment_means", floats([[6, 7]]), ints([2]), 3, **TOL)


# means_scatter_overwrites is right on the first input: no segment has two rows.
@case("segment_means: every row is its own segment, N = S = 4, and N = D = S = 4")
def means_one_row_each(ctx):
    x = floats([[1, 10], [3, 30], [5, 50], [7, 70]])
    got = ctx.check("segment_means", x, ints([2, 0, 3, 1]), 4, **TOL)
    check.allclose(got, x[[1, 3, 0, 2]], label="segment_means")
    ctx.check("segment_means", ctx.uniform(4, 4, low=-2.0, high=2.0), ints([3, 3, 0, 3]), 4, **TOL)


@case("segment_means: random values, N = 64, D = 5, S = 8, and all rows in one of 16 segments")
def means_random(ctx):
    for _ in range(3):
        ctx.check("segment_means", *seg_inputs(ctx, 64, 5, 8), **TOL)
    x = ctx.uniform(64, 5, low=-2.0, high=2.0)
    ctx.check("segment_means", x, np.full(64, 5, dtype=np.int64), 16, **TOL)


@case("segment_means: one feature, D = 1, and N = D = S = 1")
def means_size_one(ctx):
    ctx.check("segment_means", floats([[1], [2], [6], [10]]), ints([1, 1, 0, 1]), 3, **TOL)
    ctx.check("segment_means", floats([[5]]), ints([0]), 1, **TOL)


def speed_rows(ctx, rows):
    """The row count of a speed case: a quarter in JAX, where a row loop is slowest."""
    return rows // 4 if ctx.framework == "jax" else rows


@case("segment_means: the size of the speed case, N = 4096 (1024 in JAX), D = 16, S = 8")
def means_large(ctx):
    ctx.check("segment_means", *seg_inputs(ctx, speed_rows(ctx, 4096), 16, 8), **TOL)


# The row_loops mutant measures more than 75x here. Not run for MLX: see the table at
# the top of this file.
@case(
    "segment_means: speed, within 20x of the reference for N = 4096 (1024 in JAX), D = 16, S = 8",
    public=True,
    frameworks=("torch", "jax"),
    time_limit_s=30,
)
def means_speed(ctx):
    inputs = seg_inputs(ctx, speed_rows(ctx, 4096), 16, 8)
    ctx.check_speed("segment_means", *inputs, max_ratio=20.0, **TIMING)


# ----------------------------------------------------------------- common_prefix_len


# Rejects prefix_counts_all_matches (row 0 matches again at position 3: 3 for 2) and
# prefix_by_argmax (row 1 is equal: 0 for 4).
@case("common_prefix_len: the worked example, prefixes of 2, 4, 0, and 3 out of T = 4", public=True)
def prefix_example(ctx):
    a = ints([[5, 6, 7, 8], [5, 6, 7, 8], [1, 2, 3, 4], [9, 9, 9, 9]])
    b = ints([[5, 6, 0, 8], [5, 6, 7, 8], [0, 2, 3, 4], [9, 9, 9, 0]])
    got = ctx.check("common_prefix_len", a, b)
    check.equal(got, ints([2, 4, 0, 3]), label="common_prefix_len")


# Rejects prefix_bare_squeeze: the result must be (1,), and squeeze() returns a tensor
# with no axes.
@case("common_prefix_len: a batch of one returns shape (1,)", public=True)
def prefix_batch_of_one(ctx):
    got = ctx.check("common_prefix_len", ints([[3, 1, 4, 1, 5]]), ints([[3, 1, 4, 2, 5]]))
    check.equal(got, ints([3]), label="common_prefix_len")


@case("common_prefix_len: equal rows give T = 5 for every row", public=True)
def prefix_equal_rows(ctx):
    a = ctx.randint(0, 1000, (3, 5))
    got = ctx.check("common_prefix_len", a, a.copy())
    check.equal(got, ints([5, 5, 5]), label="common_prefix_len")


@case("common_prefix_len: as many rows as positions, B = T = 4, a mismatch at each position")
def prefix_square(ctx):
    a = np.tile(np.arange(4, dtype=np.int64), (4, 1))
    b = a.copy()
    b[np.arange(4), np.arange(4)] = 99  # row i first differs at position i
    b[0, 2] = 99  # and row 0 again later
    got = ctx.check("common_prefix_len", a, b)
    check.equal(got, ints([0, 1, 2, 3]), label="common_prefix_len")


@case("common_prefix_len: random ids from a vocabulary of 2 and of 3, B = 6, T = 7")
def prefix_random(ctx):
    for vocab in (2, 3, 2):
        ctx.check("common_prefix_len", ctx.randint(0, vocab, (6, 7)), ctx.randint(0, vocab, (6, 7)))


@case("common_prefix_len: a single position, equal and not, and a (1, 1) pair")
def prefix_size_one(ctx):
    got = ctx.check("common_prefix_len", ints([[4], [5], [6]]), ints([[4], [0], [6]]))
    check.equal(got, ints([1, 0, 1]), label="common_prefix_len")
    ctx.check("common_prefix_len", ints([[4]]), ints([[4]]))
    ctx.check("common_prefix_len", ints([[4]]), ints([[5]]))


# The row_loops mutant measures more than 100x here.
@case(
    "common_prefix_len: speed, within 20x of the reference for B = 2048, T = 16",
    public=True,
    time_limit_s=30,
)
def prefix_speed(ctx):
    a = ctx.randint(0, 2, (2048, 16))
    b = ctx.randint(0, 2, (2048, 16))
    ctx.check("common_prefix_len", a, b)
    ctx.check_speed("common_prefix_len", a, b, max_ratio=20.0, **TIMING)


# --------------------------------------------------------------- repeated_bigram_ban


# Rejects ban_wraps_past_the_end (row 2 would ban its first token, 0) and
# ban_last_token_itself (rows 0 and 2 would ban their last token).
@case("repeated_bigram_ban: the worked example, 4 rows of 6 tokens, V = 6", public=True)
def ban_example(ctx):
    ids = ints([[1, 2, 3, 1, 4, 1], [3, 3, 0, 1, 2, 3], [0, 1, 2, 3, 4, 5], [2, 2, 2, 2, 2, 2]])
    got = ctx.check("repeated_bigram_ban", ids, 6)
    check.equal(got, bools(["..#.#.", "#..#..", "......", "..#..."]), label="repeated_bigram_ban")


# Rejects ban_ignores_final_pair: the pair (4, 4) ends at the last position.
@case("repeated_bigram_ban: a row ending in the same token twice bans that token", public=True)
def ban_final_pair(ctx):
    got = ctx.check("repeated_bigram_ban", ints([[0, 1, 4, 4], [0, 1, 2, 4]]), 5)
    check.equal(got, bools(["....#", "....."]), label="repeated_bigram_ban")


# Rejects ban_setting_scatter whichever write it keeps: 2 follows the last token first
# and another token later, 4 the other way around.
@case(
    "repeated_bigram_ban: a token that follows the last token once and another token once",
    public=True,
)
def ban_mixed_followers(ctx):
    got = ctx.check("repeated_bigram_ban", ints([[1, 2, 3, 2, 5, 4, 1, 4, 1]]), 6)
    check.equal(got, bools(["..#.#."]), label="repeated_bigram_ban")


@case("repeated_bigram_ban: a row of one token has no pair and bans nothing", public=True)
def ban_one_token(ctx):
    got = ctx.check("repeated_bigram_ban", ints([[2], [0]]), 4)
    check.equal(got, bools(["....", "...."]), label="repeated_bigram_ban")


@case("repeated_bigram_ban: equal sizes, B = T = V = 4, and random ids of shape (5, 7)")
def ban_square(ctx):
    ids = ints([[0, 1, 0, 2], [3, 3, 3, 3], [1, 0, 1, 1], [2, 3, 0, 1]])
    got = ctx.check("repeated_bigram_ban", ids, 4)
    check.equal(got, bools(["....", "...#", "##..", "...."]), label="repeated_bigram_ban")
    for vocab in (2, 3, 5):
        ctx.check("repeated_bigram_ban", ctx.randint(0, vocab, (5, 7)), vocab)


@case("repeated_bigram_ban: a larger vocabulary than any token used, V = 16, T = 2 and T = 3")
def ban_short_rows(ctx):
    ctx.check("repeated_bigram_ban", ints([[7, 7], [7, 8]]), 16)
    ctx.check("repeated_bigram_ban", ints([[9, 3, 9], [9, 9, 9], [3, 9, 9]]), 16)


@case("repeated_bigram_ban: a vocabulary of one token, and a (1, 1) input")
def ban_size_one(ctx):
    got = ctx.check("repeated_bigram_ban", ints([[0, 0, 0]]), 1)
    check.equal(got, bools(["#"]), label="repeated_bigram_ban")
    got = ctx.check("repeated_bigram_ban", ints([[0]]), 1)
    check.equal(got, bools(["."]), label="repeated_bigram_ban")


# The row_loops mutant measures more than 150x here. The slowest correct solution,
# paired indexing through torch.unique, measures 12x in PyTorch.
@case(
    "repeated_bigram_ban: speed, within 20x of the reference for B = 2048 (512 in JAX), T = V = 16",
    public=True,
    time_limit_s=30,
)
def ban_speed(ctx):
    ids = ctx.randint(0, 16, (speed_rows(ctx, 2048), 16))
    ctx.check("repeated_bigram_ban", ids, 16)
    ctx.check_speed("repeated_bigram_ban", ids, 16, max_ratio=20.0, **TIMING)
