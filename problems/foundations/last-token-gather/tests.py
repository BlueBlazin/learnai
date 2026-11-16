"""Tests for last-token-gather. Selection copies values, so every comparison is exact."""

import numpy as np

from learnai.judge import case, check

FN = "last_token_logits"
EXACT = {"rtol": 0.0, "atol": 0.0}
SIDES = ("right", "left")


def lengths_of(*values):
    return np.array(values, dtype=np.int64)


def labeled(batch, seq, vocab):
    """logits[b, t, v] = 100 b + 10 t + v: every entry names its own position."""
    b, t, v = np.meshgrid(np.arange(batch), np.arange(seq), np.arange(vocab), indexing="ij")
    return (100 * b + 10 * t + v).astype(np.float32)


@case("the worked example: lengths [3, 1], right padding", public=True)
def worked_example_right(ctx):
    got = ctx.check(FN, labeled(2, 3, 2), lengths_of(3, 1), "right", **EXACT)
    expected = np.array([[20.0, 21.0], [100.0, 101.0]], dtype=np.float32)
    check.allclose(got, expected, label=FN, **EXACT)


@case("the worked example: lengths [3, 1], left padding", public=True)
def worked_example_left(ctx):
    got = ctx.check(FN, labeled(2, 3, 2), lengths_of(3, 1), "left", **EXACT)
    expected = np.array([[20.0, 21.0], [120.0, 121.0]], dtype=np.float32)
    check.allclose(got, expected, label=FN, **EXACT)


# No row is full, so length_as_index reads a wrong (padding) position instead of
# running off the end. Rejects always_last_column and shared_max_length as well.
@case("a right-padded batch of mixed lengths: B = 4, T = 6, V = 8", public=True)
def right_padded(ctx):
    ctx.check(FN, ctx.randn(4, 6, 8, scale=3.0), lengths_of(2, 5, 1, 3), "right", **EXACT)


# Rejects ignores_padding_side and left_first_real_token.
@case("a left-padded batch of mixed lengths: B = 4, T = 6, V = 8", public=True)
def left_padded(ctx):
    ctx.check(FN, ctx.randn(4, 6, 8, scale=3.0), lengths_of(2, 5, 1, 3), "left", **EXACT)


@case("no padding: every row is full, so the answer is the last column", public=True)
def no_padding(ctx):
    logits = ctx.randn(4, 6, 8, scale=3.0)
    for side in SIDES:
        got = ctx.check(FN, logits, lengths_of(6, 6, 6, 6), side, **EXACT)
        check.allclose(got, logits[:, -1], label=FN, **EXACT)


# Rejects onehot_multiply: the selected values are finite, other positions hold -inf.
@case("logits that contain -inf: banned tokens, and whole padding positions")
def infinite_logits(ctx):
    logits = ctx.randn(4, 6, 8, scale=3.0)
    logits[:, :, 2] = -np.inf  # a token banned at every position
    lengths = lengths_of(2, 5, 1, 3)
    right = logits.copy()
    for b, n in enumerate(lengths):
        right[b, n:] = -np.inf  # padding positions hold nothing meaningful
    ctx.check(FN, right, lengths, "right", **EXACT)
    left = logits.copy()
    for b, n in enumerate(lengths):
        left[b, : 6 - n] = -np.inf
    ctx.check(FN, left, lengths, "left", **EXACT)


# Rows of equal length below T: shared_max_length is right here, always_last_column is
# not. The mixed batches above tell the first one apart.
@case("all rows the same length, shorter than T")
def equal_lengths(ctx):
    logits = ctx.randn(4, 6, 8, scale=3.0)
    for side in SIDES:
        ctx.check(FN, logits, lengths_of(3, 3, 3, 3), side, **EXACT)


# Rejects gathers_vocab_axis, whose (B, T) result has the right shape here.
@case("as many positions as vocabulary entries: T = V = 6")
def seq_equals_vocab(ctx):
    for side in SIDES:
        ctx.check(FN, labeled(4, 6, 6), lengths_of(4, 1, 6, 2), side, **EXACT)
        ctx.check(FN, ctx.randn(4, 6, 6), lengths_of(4, 1, 6, 2), side, **EXACT)


# Rejects swaps_index_roles, which is silent when positions are valid row numbers.
@case("all axes the same length: B = T = V = 6")
def all_axes_equal(ctx):
    for side in SIDES:
        ctx.check(FN, labeled(6, 6, 6), lengths_of(3, 1, 6, 2, 5, 4), side, **EXACT)
        ctx.check(FN, ctx.randn(6, 6, 6), lengths_of(2, 2, 5, 6, 1, 3), side, **EXACT)


@case("a batch of one, a single position, and a vocabulary of one")
def degenerate_shapes(ctx):
    for side in SIDES:
        ctx.check(FN, ctx.randn(1, 6, 8), lengths_of(4), side, **EXACT)
        ctx.check(FN, ctx.randn(4, 1, 8), lengths_of(1, 1, 1, 1), side, **EXACT)
        ctx.check(FN, ctx.randn(4, 6, 1), lengths_of(6, 1, 3, 2), side, **EXACT)


@case("more rows than positions: B = 6, T = 3")
def more_rows_than_positions(ctx):
    for side in SIDES:
        ctx.check(FN, ctx.randn(6, 3, 8), lengths_of(1, 3, 2, 2, 1, 3), side, **EXACT)


@case("random lengths, both sides: B = 4, T = 16, V = 8")
def random_sweep(ctx):
    for _ in range(2):
        for side in SIDES:
            ctx.check(FN, ctx.randn(4, 16, 8, scale=5.0), ctx.randint(1, 17, (4,)), side, **EXACT)
