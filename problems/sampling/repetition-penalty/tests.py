"""Tests for repetition-penalty.

Numeric cases compare the solution with the oracle. The speed case times it against the
reference of the same framework on a large batch: a Python loop over rows or tokens is
tens to hundreds of times slower than one batched scatter.
"""

import numpy as np

from learnai.judge import case, check

FN = "apply_repetition_penalty"


def history(ctx, batch, length, vocab, pad_id, pad_fraction=0.3):
    """Random token histories in which about `pad_fraction` of the entries are padding."""
    tokens = ctx.randint(0, vocab, (batch, length))
    tokens[ctx.rng.random((batch, length)) < pad_fraction] = pad_id
    return tokens


@case("the worked example: two rows, penalty 1.25, pad id 0", public=True)
def worked_example(ctx):
    logits = np.array(
        [[2.0, -1.0, 0.5, -3.0, 0.0, 4.0], [1.0, 1.0, -2.0, -2.0, 3.0, -0.5]], dtype=np.float32
    )
    prev_tokens = np.array([[5, 1, 5, 0], [2, 4, 0, 0]])
    got = ctx.call(FN, logits, prev_tokens, 1.25, 0)
    expected = np.array(
        [[2.0, -1.25, 0.5, -3.0, 0.0, 3.2], [1.0, 1.0, -2.5, -2.0, 2.4, -0.5]], dtype=np.float32
    )
    check.allclose(got, expected, label=FN, dtype=True)


@case("a random batch with right-padded histories of different lengths", public=True)
def random_batch(ctx):
    logits = ctx.randn(4, 10, scale=3.0)
    prev_tokens = ctx.randint(1, 10, (4, 6))
    for row, length in enumerate([6, 4, 1, 2]):
        prev_tokens[row, length:] = 0
    ctx.check(FN, logits, prev_tokens, 1.3, 0)


@case("a token that occurs many times is penalized once", public=True)
def repeated_tokens(ctx):
    # Catches a penalty applied per occurrence (penalty ** count).
    logits = ctx.randn(4, 10, scale=3.0)
    prev_tokens = np.array(
        [[3, 3, 3, 3, 3, 3], [1, 6, 1, 6, 1, 6], [7, 2, 2, 7, 7, 7], [9, 9, 0, 0, 9, 4]]
    )
    ctx.check(FN, logits, prev_tokens, 1.5, 0)


@case("speed: within 8x of the vectorized reference for B = 4096, T = 16, V = 256", public=True)
def fast_enough(ctx):
    # The loop-over-rows mutant measures about 100x here; a correct vectorized solution
    # built from different operations stays below 4x in every framework.
    logits = ctx.randn(4096, 256, scale=3.0)
    prev_tokens = history(ctx, 4096, 16, 256, pad_id=0)
    ctx.check_speed(FN, logits, prev_tokens, 1.2, 0, max_ratio=8.0)


@case("padding is never penalized, whatever its position and whatever the pad id")
def padding_ignored(ctx):
    # Catches treating pad entries as history: the pad logit is positive in some rows
    # and negative in others, and padding sits at the start, the middle, and the end.
    for pad_id in (0, 5, 9):
        logits = ctx.randn(4, 10, scale=3.0)
        logits[:, pad_id] = np.array([4.0, -4.0, 2.5, -0.5], dtype=np.float32)
        ctx.check(FN, logits, history(ctx, 4, 6, 10, pad_id, pad_fraction=0.4), 1.7, pad_id)


@case("rows whose history is all padding are returned unchanged")
def empty_histories(ctx):
    logits = ctx.randn(4, 10, scale=3.0)
    prev_tokens = history(ctx, 4, 6, 10, pad_id=2)
    prev_tokens[1] = prev_tokens[3] = 2
    ctx.check(FN, logits, prev_tokens, 1.4, 2)
    ctx.check(FN, logits, np.full((4, 6), 2), 1.4, 2)


@case("rows have different histories: a square batch (B = V)")
def per_row_histories(ctx):
    # Catches a set of seen tokens shared by the whole batch, and row/column mix-ups.
    logits = ctx.randn(8, 8, scale=3.0)
    prev_tokens = np.stack([np.array([row, (row + 3) % 8, 0]) for row in range(8)])
    ctx.check(FN, logits, prev_tokens, 1.6, 0)


@case("zero and masked (-inf) logits, and a penalty of exactly 1")
def special_values(ctx):
    logits = ctx.randn(4, 10, scale=3.0)
    logits[0, 1] = logits[1, 4] = logits[3, 2] = 0.0
    logits[0, 3] = logits[2, 5] = logits[3, 8] = -np.inf
    prev_tokens = np.array(
        [[1, 3, 6, 0, 0, 0], [4, 4, 2, 0, 0, 0], [5, 7, 0, 0, 0, 0], [0, 0, 0, 0, 0, 9]]
    )
    ctx.check(FN, logits, prev_tokens, 1.8, 0)
    ctx.check(FN, logits, prev_tokens, 1.0, 0)


@case("a history longer than the vocabulary, a single row, and a history of length 1")
def extreme_shapes(ctx):
    ctx.check(FN, ctx.randn(3, 6, scale=3.0), history(ctx, 3, 40, 6, pad_id=1), 1.2, 1)
    ctx.check(FN, ctx.randn(1, 20, scale=3.0), history(ctx, 1, 9, 20, pad_id=0), 1.2, 0)
    ctx.check(FN, ctx.randn(4, 10, scale=3.0), np.array([[3], [0], [9], [3]]), 2.0, 0)


@case("a larger random batch: B = 64, T = 48, V = 1000")
def larger_batch(ctx):
    logits = ctx.randn(64, 1000, scale=4.0)
    ctx.check(FN, logits, history(ctx, 64, 48, 1000, pad_id=3), 1.15, 3)
