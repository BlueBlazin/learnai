"""Tests for masks-and-where.

Every case names the function it tests. Unless a case says otherwise, some of the
entries that the mask drops are overwritten with inf, -inf, and NaN: a solution that
selects never reads them, and one that multiplies by the mask turns them into NaN.
"""

import numpy as np

from learnai.judge import case, check

EXACT = {"rtol": 0.0, "atol": 0.0}  # fill_masked and masked_max copy values
INF = np.float32(np.inf)
NAN = np.float32(np.nan)
POISON = np.array([np.inf, -np.inf, np.nan], dtype=np.float32)


def labeled(batch, seq, width):
    """x[b, t, d] = 100 b + 10 t + d: every entry names its own position."""
    b, t, d = np.meshgrid(np.arange(batch), np.arange(seq), np.arange(width), indexing="ij")
    return (100 * b + 10 * t + d).astype(np.float32)


def bools(values):
    return np.array(values, dtype=bool)


def poisoned(ctx, x, kept):
    """A copy of x with about half of the entries that are not kept set to inf, -inf, or NaN.

    kept has the shape of x, or lacks its last axis (a (B, T) mask for a (B, T, D) x).
    """
    if kept.ndim < x.ndim:
        kept = kept[..., None]
    hit = np.broadcast_to(~kept, x.shape) & (ctx.rng.random(x.shape) < 0.5)
    out = x.copy()
    out[hit] = POISON[ctx.rng.integers(0, 3, size=int(hit.sum()))]
    return out


def random_mask(ctx, *shape):
    return ctx.rng.random(shape) < 0.5


EXAMPLE_X = np.array(
    [
        [[0.0, 1.0], [INF, NAN], [20.0, 21.0], [30.0, -INF]],
        [[100.0, 101.0], [110.0, 111.0], [120.0, 121.0], [130.0, 131.0]],
        [[200.0, 201.0], [NAN, 211.0], [220.0, -INF], [230.0, 231.0]],
    ],
    dtype=np.float32,
)
EXAMPLE_MASK = bools(
    [[True, False, True, False], [False, False, False, True], [False, False, False, False]]
)


# ----------------------------------------------------------------------- fill_masked


# Rejects fill_large_negative (-1e9 where -inf is expected), fill_by_adding_mask (NaN
# where the dropped score is inf or NaN), and fill_inverted.
@case("fill_masked: the worked example, dropped scores hold inf, NaN, and -inf", public=True)
def fill_example(ctx):
    scores = np.array([[1.0, 2.0, INF, 4.0], [NAN, -1.0, 0.5, -INF]], dtype=np.float32)
    keep = bools([[True, True, False, False], [False, True, True, False]])
    got = ctx.check("fill_masked", scores, keep, **EXACT)
    expected = np.array([[1.0, 2.0, -INF, -INF], [-INF, -1.0, 0.5, -INF]], dtype=np.float32)
    check.allclose(got, expected, label="fill_masked", **EXACT)


@case(
    "fill_masked: a row that keeps nothing is all -inf, a row that keeps all is unchanged",
    public=True,
)
def fill_none_and_all(ctx):
    scores = ctx.randn(2, 4)
    keep = bools([[False] * 4, [True] * 4])
    got = ctx.check("fill_masked", scores, keep, **EXACT)
    check.allclose(got[1], scores[1], label="fill_masked, the row that keeps everything", **EXACT)


@case("fill_masked: a square input, B = V = 3, and a random (4, 6) input")
def fill_random(ctx):
    for shape in ((3, 3), (4, 6), (4, 6)):
        keep = random_mask(ctx, *shape)
        ctx.check("fill_masked", poisoned(ctx, ctx.randn(*shape), keep), keep, **EXACT)


@case("fill_masked: a batch of one, a single column, and a (1, 1) input kept and dropped")
def fill_size_one(ctx):
    keep = bools([[True, False, False, True]])
    ctx.check("fill_masked", poisoned(ctx, ctx.randn(1, 4), keep), keep, **EXACT)
    keep = bools([[True], [False], [False], [True]])
    ctx.check("fill_masked", poisoned(ctx, ctx.randn(4, 1), keep), keep, **EXACT)
    ctx.check("fill_masked", np.array([[7.0]], dtype=np.float32), bools([[True]]), **EXACT)
    ctx.check("fill_masked", np.array([[NAN]], dtype=np.float32), bools([[False]]), **EXACT)


# ----------------------------------------------------------------------- masked_mean


# Rejects mean_multiplies_by_mask (row 0 is NaN: its dropped step holds inf and NaN),
# mean_divides_by_length (row 1: 32.5 for 130), mean_nan_for_empty_row (row 2), and
# mean_count_not_aligned (it raises: 3 rows against 2 features).
@case(
    "masked_mean: the worked example, x of shape (3, 4, 2) with a row that keeps nothing",
    public=True,
)
def mean_example(ctx):
    got = ctx.check("masked_mean", EXAMPLE_X, EXAMPLE_MASK)
    expected = np.array([[10.0, 11.0], [130.0, 131.0], [0.0, 0.0]], dtype=np.float32)
    check.allclose(got, expected, label="masked_mean")


# Rejects mean_divides_by_length on input with nothing but ordinary numbers.
@case("masked_mean: every dropped step holds an ordinary number, x of shape (2, 4, 3)", public=True)
def mean_finite(ctx):
    mask = bools([[True, True, False, True], [False, True, False, False]])
    ctx.check("masked_mean", labeled(2, 4, 3), mask)


# Rejects mean_bare_squeeze: the result must be (1, 2), and squeeze() returns (2,).
@case("masked_mean: a batch of one keeps its batch axis, x of shape (1, 4, 2)", public=True)
def mean_batch_of_one(ctx):
    mask = bools([[False, True, True, False]])
    ctx.check("masked_mean", poisoned(ctx, labeled(1, 4, 2), mask), mask)


# Rejects mean_bare_squeeze: the result must be (2, 1), and squeeze() returns (2,).
@case("masked_mean: a single feature keeps its feature axis, x of shape (2, 4, 1)", public=True)
def mean_one_feature(ctx):
    mask = bools([[True, False, True, True], [False, False, True, False]])
    ctx.check("masked_mean", poisoned(ctx, labeled(2, 4, 1), mask), mask)


# Rejects mean_count_not_aligned with wrong values: the (3,) counts line up with the
# features, and the rows keep 1, 2, and 4 steps.
@case("masked_mean: as many rows as features, B = D = 3, T = 4", public=True)
def mean_square(ctx):
    mask = bools([[False, False, True, False], [True, False, False, True], [True] * 4])
    ctx.check("masked_mean", poisoned(ctx, labeled(3, 4, 3), mask), mask)


@case("masked_mean: all sizes equal, B = T = D = 3, and a mask that keeps every step")
def mean_all_equal(ctx):
    mask = bools([[True, False, False], [False, True, True], [False, False, False]])
    ctx.check("masked_mean", poisoned(ctx, labeled(3, 3, 3), mask), mask)
    x = ctx.randn(3, 3, 3)
    got = ctx.check("masked_mean", x, np.ones((3, 3), dtype=bool))
    check.allclose(got, x.mean(axis=1), label="masked_mean with every step kept")


@case("masked_mean: random masks, B = 4, T = 5, D = 3")
def mean_random(ctx):
    for _ in range(3):
        mask = random_mask(ctx, 4, 5)
        ctx.check("masked_mean", poisoned(ctx, ctx.randn(4, 5, 3, scale=10.0), mask), mask)


@case("masked_mean: a single step, kept and dropped, and all sizes 1")
def mean_size_one(ctx):
    mask = bools([[True], [False]])
    ctx.check("masked_mean", poisoned(ctx, labeled(2, 1, 3), mask), mask)
    ctx.check("masked_mean", np.array([[[7.0]]], dtype=np.float32), bools([[True]]))
    ctx.check("masked_mean", np.array([[[INF]]], dtype=np.float32), bools([[False]]))


# ------------------------------------------------------------------------ masked_max


# Rejects max_over_product (row 0 is NaN, row 2 is 0 for -inf) and
# max_zero_for_empty_row (row 2).
@case(
    "masked_max: the worked example, x of shape (3, 4, 2) with a row that keeps nothing",
    public=True,
)
def max_example(ctx):
    got = ctx.check("masked_max", EXAMPLE_X, EXAMPLE_MASK, **EXACT)
    expected = np.array([[20.0, 21.0], [130.0, 131.0], [-INF, -INF]], dtype=np.float32)
    check.allclose(got, expected, label="masked_max", **EXACT)


# Rejects max_over_product on input with nothing but ordinary numbers: a dropped step
# counts as 0 there, and 0 is larger than every kept value.
@case("masked_max: every kept value is negative, every dropped one an ordinary number", public=True)
def max_negative(ctx):
    x = -1.0 - labeled(2, 4, 3)
    mask = bools([[False, True, False, True], [False, False, True, False]])
    ctx.check("masked_max", x, mask, **EXACT)


@case("masked_max: a batch of one, x of shape (1, 4, 2)", public=True)
def max_batch_of_one(ctx):
    mask = bools([[False, True, True, False]])
    ctx.check("masked_max", poisoned(ctx, labeled(1, 4, 2), mask), mask, **EXACT)


# Rejects max_mask_on_feature_axis with wrong values: (B, 1, T) lines up with (B, T, D).
@case("masked_max: as many steps as features, T = D = 3, and B = T = D = 3", public=True)
def max_equal_axes(ctx):
    mask = bools([[True, False, False], [False, True, True]])
    ctx.check("masked_max", labeled(2, 3, 3), mask, **EXACT)
    mask = bools([[True, False, False], [False, True, True], [False, False, False]])
    ctx.check("masked_max", poisoned(ctx, labeled(3, 3, 3), mask), mask, **EXACT)


@case("masked_max: random masks, B = 4, T = 5, D = 3")
def max_random(ctx):
    for _ in range(3):
        mask = random_mask(ctx, 4, 5)
        ctx.check("masked_max", poisoned(ctx, ctx.randn(4, 5, 3, scale=10.0), mask), mask, **EXACT)


@case("masked_max: a single feature, a single step kept and dropped, and all sizes 1")
def max_size_one(ctx):
    mask = bools([[True, False, True, True], [False, False, True, False]])
    ctx.check("masked_max", poisoned(ctx, labeled(2, 4, 1), mask), mask, **EXACT)
    mask = bools([[True], [False]])
    ctx.check("masked_max", poisoned(ctx, labeled(2, 1, 3), mask), mask, **EXACT)
    ctx.check("masked_max", np.array([[[7.0]]], dtype=np.float32), bools([[True]]), **EXACT)
    ctx.check("masked_max", np.array([[[NAN]]], dtype=np.float32), bools([[False]]), **EXACT)
