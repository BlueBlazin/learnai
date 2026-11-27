"""Tests for expand-and-repeat.

Every case names the function it tests. All three functions copy values, so every
comparison is exact. A result is compared by value: a view of the input (an expanded
tensor) and a copy are both accepted. Integer inputs are passed as numpy int64 and
arrive as the framework's default integer type.
"""

import numpy as np

from learnai.judge import case, check

EXACT = {"rtol": 0.0, "atol": 0.0}


def ints(values):
    return np.array(values, dtype=np.int64)


def labeled_heads(batch, groups, seq, width):
    """k[b, g, t, d] = 1000 b + 100 g + 10 t + d: every entry names its own position."""
    b, g, t, d = np.meshgrid(
        np.arange(batch), np.arange(groups), np.arange(seq), np.arange(width), indexing="ij"
    )
    return (1000 * b + 100 * g + 10 * t + d).astype(np.float32)


def labeled_rows(batch, seq):
    """x[b, t] = 10 b + t."""
    return 10 * np.arange(batch)[:, None] + np.arange(seq)[None, :]


# ------------------------------------------------------------------- repeat_kv_heads


# Rejects heads_tiled and heads_new_axis_before_groups: heads 1 and 2 would be swapped.
# Rejects heads_repeat_time_axis (shape (1, 2, 2, 2)).
@case("repeat_kv_heads: the worked example, 2 groups repeated twice", public=True)
def heads_example(ctx):
    k = np.array([[[[0, 1]], [[100, 101]]]], dtype=np.float32)
    got = ctx.check("repeat_kv_heads", k, 2, **EXACT)
    expected = np.array([[[[0, 1]], [[0, 1]], [[100, 101]], [[100, 101]]]], dtype=np.float32)
    check.allclose(got, expected, label="repeat_kv_heads", **EXACT)


@case("repeat_kv_heads: all sizes differ, B = 2, G = 3, T = 4, D = 5, n_rep = 2", public=True)
def heads_distinct(ctx):
    got = ctx.check("repeat_kv_heads", labeled_heads(2, 3, 4, 5), 2, **EXACT)
    # head 3 is group 1: the hundreds digit of every entry is 1
    check.allclose(got[0, 3, 0], np.array([100, 101, 102, 103, 104]), label="head 3", **EXACT)


@case("repeat_kv_heads: as many copies as groups, G = n_rep = 3", public=True)
def heads_square(ctx):
    ctx.check("repeat_kv_heads", labeled_heads(2, 3, 4, 5), 3, **EXACT)


@case("repeat_kv_heads: n_rep = 1 returns k, and one group gives n_rep equal heads")
def heads_trivial(ctx):
    k = labeled_heads(2, 3, 4, 5)
    got = ctx.check("repeat_kv_heads", k, 1, **EXACT)
    check.allclose(got, k, label="repeat_kv_heads with n_rep = 1", **EXACT)
    ctx.check("repeat_kv_heads", labeled_heads(2, 1, 4, 5), 3, **EXACT)


@case("repeat_kv_heads: all sizes equal, B = G = T = D = n_rep = 2, and random values")
def heads_all_equal(ctx):
    ctx.check("repeat_kv_heads", labeled_heads(2, 2, 2, 2), 2, **EXACT)
    ctx.check("repeat_kv_heads", ctx.randn(2, 2, 2, 2), 2, **EXACT)
    ctx.check("repeat_kv_heads", ctx.randn(2, 3, 4, 5), 4, **EXACT)


@case("repeat_kv_heads: the largest sizes, B = G = T = D = 8 and n_rep = 4")
def heads_largest(ctx):
    ctx.check("repeat_kv_heads", ctx.randn(8, 8, 8, 8), 4, **EXACT)


@case("repeat_kv_heads: a batch of one, a single position, a single feature, and all sizes 1")
def heads_size_one(ctx):
    ctx.check("repeat_kv_heads", labeled_heads(1, 3, 4, 5), 2, **EXACT)
    ctx.check("repeat_kv_heads", labeled_heads(2, 3, 1, 5), 2, **EXACT)
    ctx.check("repeat_kv_heads", labeled_heads(2, 3, 4, 1), 2, **EXACT)
    ctx.check("repeat_kv_heads", labeled_heads(1, 1, 1, 1), 1, **EXACT)


# ------------------------------------------------------------------ expand_for_beams


# Rejects beams_tiled (rows 1 and 2 would be swapped) and beams_no_dim (shape (12,)).
@case("expand_for_beams: the worked example, 2 rows and K = 2", public=True)
def beams_example(ctx):
    got = ctx.check("expand_for_beams", ints([[1, 2, 3], [4, 5, 6]]), 2)
    check.equal(got, ints([[1, 2, 3], [1, 2, 3], [4, 5, 6], [4, 5, 6]]), label="expand_for_beams")


@case("expand_for_beams: a batch of one becomes K = 3 equal rows", public=True)
def beams_batch_of_one(ctx):
    got = ctx.check("expand_for_beams", ints([[7, 8, 9, 10]]), 3)
    check.shape(got, (3, 4), label="expand_for_beams")


@case("expand_for_beams: as many beams as rows, B = K = 3, T = 4", public=True)
def beams_square(ctx):
    got = ctx.check("expand_for_beams", labeled_rows(3, 4), 3)
    check.equal(got[:, 0], ints([0, 0, 0, 10, 10, 10, 20, 20, 20]), label="first column")


@case("expand_for_beams: K = 1 returns x, and all sizes equal, B = T = K = 3")
def beams_trivial(ctx):
    x = labeled_rows(3, 4)
    got = ctx.check("expand_for_beams", x, 1)
    check.equal(got, x, label="expand_for_beams with K = 1")
    ctx.check("expand_for_beams", labeled_rows(3, 3), 3)


@case("expand_for_beams: random ids, B = 5, T = 4, K = 2 and K = 4")
def beams_random(ctx):
    for beams in (2, 4):
        ctx.check("expand_for_beams", ctx.randint(0, 1000, (5, 4)), beams)


@case("expand_for_beams: the largest sizes, B = T = 8 and K = 4")
def beams_largest(ctx):
    ctx.check("expand_for_beams", ctx.randint(0, 1000, (8, 8)), 4)


# Rejects beams_bare_squeeze: the result must be (6, 1), and squeeze() returns (6,).
@case("expand_for_beams: a single position keeps its axis, T = 1, and all sizes 1", public=True)
def beams_size_one(ctx):
    got = ctx.check("expand_for_beams", ints([[5], [6], [7]]), 2)
    check.equal(got, ints([[5], [5], [6], [6], [7], [7]]), label="expand_for_beams")
    ctx.check("expand_for_beams", ints([[5]]), 1)


# -------------------------------------------------------------------- tile_positions


# Rejects positions_repeated_then_reshaped: it returns [[0, 0, 1], [1, 2, 2]].
# Rejects positions_transposed (shape (3, 2)).
@case("tile_positions: the worked example, 3 positions for a batch of 2", public=True)
def positions_example(ctx):
    got = ctx.check("tile_positions", ints([0, 1, 2]), 2)
    check.equal(got, ints([[0, 1, 2], [0, 1, 2]]), label="tile_positions")


# Rejects positions_bare_squeeze: the result must be (1, 4), and squeeze() returns (4,).
@case("tile_positions: a batch of one returns shape (1, 4)", public=True)
def positions_batch_of_one(ctx):
    got = ctx.check("tile_positions", ints([3, 4, 5, 6]), 1)
    check.equal(got, ints([[3, 4, 5, 6]]), label="tile_positions")


# positions_transposed returns the right shape here, with the positions down the columns.
@case("tile_positions: as many rows as positions, B = T = 3", public=True)
def positions_square(ctx):
    got = ctx.check("tile_positions", ints([5, 7, 9]), 3)
    check.equal(got, ints([[5, 7, 9], [5, 7, 9], [5, 7, 9]]), label="tile_positions")


@case("tile_positions: random positions, T = 6, B = 2 and B = 4")
def positions_random(ctx):
    for batch in (2, 4):
        ctx.check("tile_positions", ctx.randint(0, 1000, (6,)), batch)


@case("tile_positions: the largest sizes, T = B = 8")
def positions_largest(ctx):
    ctx.check("tile_positions", ctx.randint(0, 1000, (8,)), 8)


# Rejects positions_bare_squeeze: the result must be (4, 1), and squeeze() returns (4,).
@case("tile_positions: a single position returns shape (B, 1), and all sizes 1")
def positions_size_one(ctx):
    got = ctx.check("tile_positions", ints([8]), 4)
    check.equal(got, ints([[8], [8], [8], [8]]), label="tile_positions")
    ctx.check("tile_positions", ints([8]), 1)
