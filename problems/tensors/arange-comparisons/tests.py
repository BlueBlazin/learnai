"""Tests for arange-comparisons.

Every case names the function it tests. Every result is a boolean tensor and is
compared exactly, dtype included. Integer inputs are passed as numpy int64 and arrive
as the framework's default integer type: int64 in PyTorch, int32 in JAX and MLX.
"""

import numpy as np

from learnai.judge import case, check

SIDES = ("right", "left")


def ints(values):
    return np.array(values, dtype=np.int64)


def bools(rows):
    """A boolean array from rows written as strings: "#" is true, "." is false."""
    return np.array([[c == "#" for c in row] for row in rows], dtype=bool)


def check_lengths(ctx, lengths, T, side):
    """Compare with the oracle, then state the meaning: row b has lengths[b] true entries."""
    got = ctx.check("length_mask", lengths, T, side)
    label = f'true entries per row of length_mask(side="{side}")'
    check.equal(got.sum(axis=1), lengths, label=label)
    return got


# ----------------------------------------------------------------------- length_mask


# Rejects length_inclusive (rows 0 and 1 get one true entry too many on the right),
# length_sides_swapped, and length_left_as_complement (row 0 on the left: 1 for 3).
@case("length_mask: the worked example, lengths [3, 1, 4] with T = 4, both sides", public=True)
def length_example(ctx):
    lengths = ints([3, 1, 4])
    got = check_lengths(ctx, lengths, 4, "right")
    check.equal(got, bools(["###.", "#...", "####"]), label='length_mask(side="right")')
    got = check_lengths(ctx, lengths, 4, "left")
    check.equal(got, bools([".###", "...#", "####"]), label='length_mask(side="left")')


@case("length_mask: a row of length 0 is all false and a row of length T all true", public=True)
def length_zero_and_full(ctx):
    for side in SIDES:
        got = check_lengths(ctx, ints([0, 5]), 5, side)
        check.equal(got, bools([".....", "#####"]), label=f'length_mask(side="{side}")')


# Rejects length_bare_squeeze: the result must be (1, 4), and squeeze() returns (4,).
@case("length_mask: a batch of one returns shape (1, 4)", public=True)
def length_batch_of_one(ctx):
    for side in SIDES:
        check_lengths(ctx, ints([3]), 4, side)


# Rejects length_transposed with wrong values: (T, B) has the shape of (B, T).
@case("length_mask: as many rows as positions, B = T = 3, lengths [1, 3, 2]", public=True)
def length_square(ctx):
    for side in SIDES:
        check_lengths(ctx, ints([1, 3, 2]), 3, side)


# length_left_as_complement is right on the rows of length T / 2 and on no other.
@case("length_mask: every length from 0 to T, B = 7, T = 6")
def length_every_length(ctx):
    for side in SIDES:
        check_lengths(ctx, ints([3, 0, 6, 1, 5, 2, 4]), 6, side)


@case("length_mask: random lengths, B = 7, T = 6")
def length_random(ctx):
    for _ in range(3):
        lengths = ctx.randint(0, 7, (7,))
        for side in SIDES:
            check_lengths(ctx, lengths, 6, side)


@case("length_mask: a padded length of 1, and B = T = 1 with length 0 and 1")
def length_size_one(ctx):
    for side in SIDES:
        check_lengths(ctx, ints([1, 0, 1]), 1, side)
        check_lengths(ctx, ints([0]), 1, side)
        check_lengths(ctx, ints([1]), 1, side)


# ------------------------------------------------------------------------- band_mask


# Rejects band_no_lower_bound, band_transposed, band_one_too_wide, and band_as_float
# (dtype float32).
@case("band_mask: the worked example, T = 4 and w = 2", public=True)
def band_example(ctx):
    got = ctx.check("band_mask", 4, 2)
    check.equal(got, bools(["#...", "##..", ".##.", "..##"]), label="band_mask")


@case("band_mask: w = 1 is the diagonal alone, T = 4", public=True)
def band_width_one(ctx):
    got = ctx.check("band_mask", 4, 1)
    check.equal(got, np.eye(4, dtype=bool), label="band_mask")


# The window covers every earlier column here, so band_one_too_wide passes this case;
# band_no_lower_bound returns all true.
@case("band_mask: w = T and w > T are the whole lower triangle, T = 4", public=True)
def band_wide(ctx):
    for w in (4, 9):
        got = ctx.check("band_mask", 4, w)
        check.equal(got, np.tril(np.ones((4, 4), dtype=bool)), label=f"band_mask(4, {w})")


@case("band_mask: row i has min(i + 1, w) true entries, T = 6 and every w from 1 to 7")
def band_sweep(ctx):
    for w in range(1, 8):
        got = ctx.check("band_mask", 6, w)
        expected = np.minimum(np.arange(6) + 1, w)
        check.equal(got.sum(axis=1), expected, label=f"true entries per row of band_mask(6, {w})")


@case("band_mask: T = 1 returns [[True]] for every w")
def band_size_one(ctx):
    for w in (1, 3):
        ctx.check("band_mask", 1, w)


# ---------------------------------------------------------------------- same_segment


# Rejects segment_ids_as_positions and segment_adjacent_only (positions 0 and 3 are in
# segment 7, with segment 2 between them).
@case("same_segment: the worked example, seg [7, 7, 2, 7]", public=True)
def segment_example(ctx):
    got = ctx.check("same_segment", ints([7, 7, 2, 7]))
    check.equal(got, bools(["##.#", "##.#", "..#.", "##.#"]), label="same_segment")


@case(
    "same_segment: one segment is all true, and T different segments are the diagonal", public=True
)
def segment_extremes(ctx):
    got = ctx.check("same_segment", ints([5, 5, 5, 5]))
    check.equal(got, np.ones((4, 4), dtype=bool), label="same_segment with one segment")
    got = ctx.check("same_segment", ints([3, 0, 2, 1]))
    check.equal(got, np.eye(4, dtype=bool), label="same_segment with four segments")


# segment_ids_as_positions returns the diagonal for seg [0, 1, 2, 3], which is right;
# these ids are positions in another order, and ids that alternate.
@case("same_segment: ids that are a permutation of the positions, and alternating ids")
def segment_orders(ctx):
    ctx.check("same_segment", ints([2, 0, 3, 1]))
    ctx.check("same_segment", ints([0, 1, 0, 1, 0, 1]))


@case("same_segment: random ids, T = 6, small and large")
def segment_random(ctx):
    for high in (2, 3, 1000):
        ctx.check("same_segment", ctx.randint(0, high + 1, (6,)))


@case("same_segment: T = 1 returns [[True]]")
def segment_size_one(ctx):
    ctx.check("same_segment", ints([4]))
