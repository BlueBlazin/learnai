"""Tests for frequency-presence-penalty.

Every case compares the solution with the oracle. Counts are integers and are compared
exactly. Penalized logits are differences of numbers of order 1 to 100, so they are
compared with an absolute tolerance of 1e-5 next to the usual relative 1e-4: a correct
solution that subtracts the two penalties one after the other rounds differently from
one that adds them up first.
"""

import numpy as np

from learnai.judge import case, check

COUNT = "count_generated"
APPLY = "apply_penalties"
INF = np.inf
ATOL = 1e-5


def check_counts(ctx, tokens, prompt_lengths, vocab_size):
    tokens, prompt_lengths = np.asarray(tokens), np.asarray(prompt_lengths)
    got = ctx.call(COUNT, tokens, prompt_lengths, vocab_size)
    check.that(
        np.asarray(got).dtype.kind in "iu",
        f"{COUNT}: expected an integer tensor of counts, got dtype {np.asarray(got).dtype}",
    )
    check.equal(got, ctx.expected(COUNT, tokens, prompt_lengths, vocab_size), label=COUNT)
    return got


def check_apply(ctx, logits, counts, frequency_penalty, presence_penalty):
    return ctx.check(
        APPLY,
        np.asarray(logits, dtype=np.float32),
        np.asarray(counts),
        frequency_penalty,
        presence_penalty,
        atol=ATOL,
    )


def random_counts(ctx, batch, vocab, seen=0.4, high=6):
    """Counts in which about `seen` of the entries are positive, between 1 and `high` - 1."""
    counts = ctx.randint(1, high, (batch, vocab))
    counts[ctx.rng.random((batch, vocab)) >= seen] = 0
    return counts


@case("the worked example: counts from two rows, then both penalties", public=True)
def worked_example(ctx):
    tokens = np.array([[3, 2, 5, 5, 2, 5], [1, 1, 4, 0, 4, 4]])
    counts = ctx.call(COUNT, tokens, np.array([2, 3]), 6)
    want_counts = np.array([[0, 0, 1, 0, 0, 3], [1, 0, 0, 0, 2, 0]])
    check.equal(counts, want_counts, label=COUNT)

    logits = np.array(
        [[1.0, 0.0, 2.0, -1.0, 0.5, 3.0], [2.0, 4.0, -INF, 0.0, 1.0, -2.0]], dtype=np.float32
    )
    got = ctx.call(APPLY, logits, want_counts, 0.5, 1.0)
    expected = np.array(
        [[1.0, 0.0, 0.5, -1.0, 0.5, 0.5], [0.5, 4.0, -INF, 0.0, -1.0, -2.0]], dtype=np.float32
    )
    check.allclose(got, expected, atol=ATOL, label=APPLY, dtype=True)


@case("apply_penalties on a random batch: tokens seen 0 to 5 times", public=True)
def random_penalties(ctx):
    # Catches a multiplicative penalty and a penalty that is added.
    logits = ctx.randn(4, 10, scale=3.0)
    counts = random_counts(ctx, 4, 10)
    check_apply(ctx, logits, counts, 0.3, 0.8)
    check_apply(ctx, logits, counts, 1.2, 0.1)


@case("each penalty alone: frequency grows with the count, presence is charged once", public=True)
def one_penalty_at_a_time(ctx):
    # Catches a presence penalty multiplied by the count: with frequency_penalty = 0,
    # a token seen four times must lose exactly as much as a token seen once.
    logits = ctx.randn(3, 8, scale=2.0)
    counts = np.array(
        [[0, 1, 4, 0, 2, 1, 0, 9], [3, 0, 0, 0, 1, 1, 7, 0], [0, 0, 0, 0, 0, 0, 0, 0]]
    )
    check_apply(ctx, logits, counts, 0.0, 1.5)
    check_apply(ctx, logits, counts, 0.7, 0.0)
    check_apply(ctx, logits, counts, 0.0, 0.0)


@case("count_generated: rows with different prompt lengths and repeated tokens", public=True)
def different_prompt_lengths(ctx):
    # Catches counting the prompt, an off-by-one at the prompt boundary, one prompt
    # length used for every row, and repeated tokens counted once.
    tokens = np.array(
        [
            [4, 4, 4, 1, 1, 6, 1, 1],  # token 4 only in the prompt
            [0, 7, 7, 7, 7, 2, 7, 0],
            [5, 5, 5, 5, 5, 5, 5, 5],  # the same token in the prompt and after it
            [3, 1, 2, 6, 0, 4, 7, 5],
        ]
    )
    check_counts(ctx, tokens, [3, 1, 5, 4], 8)
    check_counts(ctx, tokens, [1, 6, 2, 7], 8)


@case("prompt lengths of 0 (everything was generated) and T (nothing was)")
def prompt_length_extremes(ctx):
    tokens = ctx.randint(0, 8, (4, 8))
    check_counts(ctx, tokens, [0, 8, 0, 8], 8)
    check_counts(ctx, tokens, [8, 8, 8, 8], 8)
    check_counts(ctx, tokens, [0, 0, 0, 0], 8)
    check_counts(ctx, tokens, [7, 1, 8, 0], 8)


@case("count_generated: a vocabulary larger than any token id, T = 1, and B = 1")
def count_shapes(ctx):
    check_counts(ctx, ctx.randint(0, 5, (4, 8)), [2, 4, 0, 7], 40)
    check_counts(ctx, np.array([[2], [0], [2]]), [0, 1, 0], 3)
    check_counts(ctx, ctx.randint(0, 12, (1, 30)), [11], 12)


@case("masked (-inf) logits stay -inf; zero and negative logits are penalized by subtraction")
def special_logits(ctx):
    logits = ctx.randn(4, 10, scale=3.0)
    logits[0, 1] = logits[1, 4] = logits[3, 2] = 0.0
    logits[0, 3] = logits[2, 5] = logits[3, 8] = -INF
    logits[2, :5] = -np.abs(logits[2, :5])
    counts = random_counts(ctx, 4, 10, seen=0.6)
    counts[0, 3] = 2  # a masked token that was generated before it was masked
    got = check_apply(ctx, logits, counts, 0.6, 0.9)
    check.that(not np.isnan(got).any(), f"{APPLY}: the result contains NaN")
    check_apply(ctx, logits, counts, 2.0, 2.0)


@case("negative penalties raise the logits of repeated tokens")
def negative_penalties(ctx):
    logits = ctx.randn(4, 10, scale=3.0)
    counts = random_counts(ctx, 4, 10)
    check_apply(ctx, logits, counts, -0.5, -1.0)
    check_apply(ctx, logits, counts, 0.4, -1.3)
    check_apply(ctx, logits, counts, -2.0, 2.0)


@case("a square batch (B = V), a single row, and large counts")
def penalty_shapes(ctx):
    check_apply(ctx, ctx.randn(6, 6, scale=2.0), random_counts(ctx, 6, 6), 0.25, 0.75)
    check_apply(ctx, ctx.randn(1, 12, scale=2.0), random_counts(ctx, 1, 12), 0.25, 0.75)
    check_apply(ctx, ctx.randn(4, 10, scale=2.0), random_counts(ctx, 4, 10, high=200), 0.1, 0.5)


@case("end to end: count the generated tokens of a batch, then penalize its logits")
def end_to_end(ctx):
    # T = V = 8 with every prompt length from 0 to 3: a square table of positions
    # against tokens, which hides a mask built along the wrong axis.
    tokens = ctx.randint(0, 8, (4, 8))
    counts = check_counts(ctx, tokens, np.array([0, 3, 1, 2]), 8)
    check_apply(ctx, ctx.randn(4, 8, scale=3.0), counts, 0.2, 0.6)


@case("a larger batch: B = 64, T = 128, V = 1000")
def larger_batch(ctx):
    tokens = ctx.randint(0, 1000, (64, 128))
    tokens[:, 40:] = tokens[:, 40:] % 50  # generated tokens repeat a lot
    counts = check_counts(ctx, tokens, ctx.randint(0, 129, (64,)), 1000)
    check_apply(ctx, ctx.randn(64, 1000, scale=4.0), counts, 0.15, 0.4)
