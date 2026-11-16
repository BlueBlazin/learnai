"""Tests for attention-mask-builder.

Most cases compare with the oracle; the mask holds only 0.0 and -inf, so the comparison
is exact. A few cases state what the mask is for (a softmax over each row) in numpy.
"""

import numpy as np

from learnai.judge import case, check

FN = "build_mask"
NINF = -np.inf
SIDES = ("right", "left")


def lengths_of(*values):
    return np.array(values, dtype=np.int64)


def softmax_rows(scores):
    """Softmax over the last axis, in float64, as an attention layer would apply it."""
    with np.errstate(invalid="ignore", over="ignore"):
        shifted = scores - scores.max(axis=-1, keepdims=True)
        e = np.exp(shifted)
        return e / e.sum(axis=-1, keepdims=True)


def real_positions(lengths, seq, side):
    pos = np.arange(seq)
    if side == "right":
        return pos[None, :] < lengths[:, None]
    return pos[None, :] >= seq - lengths[:, None]


@case("the worked example: T = 4, lengths [4, 2], right padding", public=True)
def worked_example_right(ctx):
    got = ctx.check(FN, lengths_of(4, 2), 4, "right")
    expected = np.array(
        [
            [[[0, NINF, NINF, NINF], [0, 0, NINF, NINF], [0, 0, 0, NINF], [0, 0, 0, 0]]],
            [[[0, NINF, NINF, NINF], [0, 0, NINF, NINF], [0, 0, 0, NINF], [0, 0, NINF, 0]]],
        ],
        dtype=np.float32,
    )
    check.allclose(got, expected, label=FN)


@case("the worked example: T = 4, lengths [4, 2], left padding", public=True)
def worked_example_left(ctx):
    got = ctx.check(FN, lengths_of(4, 2), 4, "left")
    expected = np.array(
        [
            [[[0, NINF, NINF, NINF], [0, 0, NINF, NINF], [0, 0, 0, NINF], [0, 0, 0, 0]]],
            [[[0, NINF, NINF, NINF], [NINF, 0, NINF, NINF], [NINF, NINF, 0, NINF], [NINF, NINF, 0, 0]]],
        ],
        dtype=np.float32,
    )
    check.allclose(got, expected, label=FN)


# Rejects sees_next_token and missing_head_axis with nothing else in the way.
@case("no padding: every row is the plain causal mask, T = 6", public=True)
def no_padding(ctx):
    for side in SIDES:
        got = ctx.check(FN, lengths_of(6, 6, 6, 6), 6, side)
        causal = np.where(np.tril(np.ones((6, 6), dtype=bool)), 0.0, NINF).astype(np.float32)
        check.allclose(got, np.broadcast_to(causal, (4, 1, 6, 6)), label=FN)


# Rejects padding_rows_all_masked with a message that names the consequence, and
# finite_negative with one that names the rule.
@case("every row keeps one open entry, and closed entries are -inf: left padding", public=True)
def rows_are_usable(ctx):
    lengths = lengths_of(1, 3, 6, 4)
    mask = ctx.call(FN, lengths, 6, "left")
    check.shape(mask, (4, 1, 6, 6), label=FN)
    check.that(
        bool(np.all((mask == 0.0) | np.isneginf(mask))),
        f"{FN}: the mask must hold only 0.0 and -inf",
        f"values found: {np.unique(mask)}",
    )
    weights = softmax_rows(ctx.randn(4, 2, 6, 6).astype(np.float64) + mask)
    check.finite(weights, label="softmax(scores + mask)")
    ctx.check(FN, lengths, 6, "left")


@case("a batch of mixed lengths, both padding sides: B = 4, T = 6", public=True)
def mixed_lengths(ctx):
    for side in SIDES:
        ctx.check(FN, lengths_of(2, 6, 4, 1), 6, side)


# States the purpose without the oracle. Rejects masks_queries, swaps_sides, and
# length_off_by_one: real queries must put no weight on padding or on the future.
@case("after softmax, real queries put zero weight on padding keys and on future keys")
def real_queries_ignore_padding(ctx):
    lengths = lengths_of(3, 1, 6, 4)
    for side in SIDES:
        mask = ctx.call(FN, lengths, 6, side)
        check.shape(mask, (4, 1, 6, 6), label=FN)
        weights = softmax_rows(ctx.randn(4, 3, 6, 6).astype(np.float64) + mask)
        real = real_positions(lengths, 6, side)  # (B, T)
        future = np.triu(np.ones((6, 6), dtype=bool), k=1)
        forbidden = (~real[:, None, None, :] | future) & real[:, None, :, None]
        leaked = np.where(np.broadcast_to(forbidden, weights.shape), weights, 0.0)
        check.that(
            float(np.nan_to_num(leaked, nan=1.0).max()) == 0.0,
            f"{FN}: a real query attends to a padding key or a future key ({side} padding)",
            f"lengths = {lengths.tolist()}, T = 6, largest leaked weight = {np.nanmax(leaked)}",
        )
        visible = np.where(np.broadcast_to(real[:, None, :, None], weights.shape), weights, 1.0)
        check.finite(visible, label="softmax(scores + mask) for real queries")


# Rejects padding_query_isolated (right padding) and padding_rows_all_masked (left).
@case("padding queries: the real keys before them and themselves, nothing else")
def padding_queries(ctx):
    ctx.check(FN, lengths_of(1, 2, 3, 4), 6, "right")
    ctx.check(FN, lengths_of(1, 2, 3, 4), 6, "left")


@case("rows of length 1 next to full rows")
def extreme_lengths(ctx):
    for side in SIDES:
        ctx.check(FN, lengths_of(1, 6, 1, 6), 6, side)


@case("a single position, T = 1, and a batch of one")
def tiny(ctx):
    for side in SIDES:
        ctx.check(FN, lengths_of(1, 1, 1, 1), 1, side)
        ctx.check(FN, lengths_of(4), 6, side)


# B == T, with a different length in every row: an axis mix-up between batch, query,
# and key cannot hide behind a non-square shape.
@case("as many rows as positions: B = T = 6, lengths 1 to 6")
def square_batch(ctx):
    for side in SIDES:
        ctx.check(FN, lengths_of(1, 2, 3, 4, 5, 6), 6, side)
        ctx.check(FN, lengths_of(6, 4, 5, 1, 3, 2), 6, side)


@case("random lengths, both sides, T = 16")
def random_sweep(ctx):
    for _ in range(2):
        for side in SIDES:
            ctx.check(FN, ctx.randint(1, 17, (4,)), 16, side)
