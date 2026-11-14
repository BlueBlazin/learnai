"""Tests for sequence-log-likelihood. Every case compares the solution with the oracle;
a few add a value that follows from the definition alone."""

import numpy as np

from whetstone.judge import case, check

BOTH = ("sequence_logprob", "perplexity")
TOL = {"rtol": 1e-4, "atol": 1e-4}  # the tolerance stated in Constraints


def padding_mask(lengths, seq, side):
    """Boolean (B, T) mask with lengths[b] real tokens on the given side."""
    pos = np.arange(seq)
    lengths = np.asarray(lengths)
    if side == "right":
        return pos[None, :] < lengths[:, None]
    return pos[None, :] >= seq - lengths[:, None]


def batch(ctx, lengths, side, *, seq=6, vocab=8, scale=2.0):
    logits = ctx.randn(len(lengths), seq, vocab, scale=scale)
    tokens = ctx.randint(0, vocab, (len(lengths), seq))
    return logits, tokens, padding_mask(lengths, seq, side)


def check_both(ctx, logits, tokens, mask):
    for fn in BOTH:
        ctx.check(fn, logits, tokens, mask, **TOL)


@case("the worked example: two rows of three tokens, one padded", public=True)
def worked_example(ctx):
    probs = np.array([[0.5, 0.25, 0.25], [0.1, 0.8, 0.1], [0.2, 0.3, 0.5]], dtype=np.float32)
    logits = np.log(np.stack([probs, probs]))
    tokens = np.array([[0, 2, 1], [1, 0, 2]], dtype=np.int64)
    mask = np.array([[True, True, True], [True, True, False]])
    logprob = ctx.check("sequence_logprob", logits, tokens, mask, **TOL)
    ppl = ctx.check("perplexity", logits, tokens, mask, **TOL)
    check.allclose(logprob, np.array([-1.609438, -0.693147], np.float32), label="sequence_logprob")
    check.allclose(ppl, np.array([2.236068, 2.0], np.float32), label="perplexity")


# Rejects no_shift and shift_backwards with no padding in the way.
@case("a random batch without padding: B = 4, T = 6, V = 8", public=True)
def no_padding(ctx):
    check_both(ctx, *batch(ctx, [6, 6, 6, 6], "right"))


# Rejects ignores_mask, mask_not_shifted, mean_over_all_positions, and exp_of_sum.
@case("a right-padded batch: lengths 2, 6, 4, 3", public=True)
def right_padded(ctx):
    check_both(ctx, *batch(ctx, [2, 6, 4, 3], "right"))


# Rejects mask_target_only: the first real token follows a padding position here.
@case("a left-padded batch: lengths 2, 6, 4, 3", public=True)
def left_padded(ctx):
    check_both(ctx, *batch(ctx, [2, 6, 4, 3], "left"))


# States perplexity without the oracle: a model that knows nothing has perplexity V.
# Rejects exp_of_sum, mean_over_all_positions, and divides_by_token_count by name.
@case("uniform logits: the perplexity is the vocabulary size, V = 8", public=True)
def uniform_logits(ctx):
    _, tokens, mask = batch(ctx, [5, 2, 6, 3], "right")
    logits = np.full((4, 6, 8), 1.5, dtype=np.float32)
    ppl = ctx.check("perplexity", logits, tokens, mask, **TOL)
    check.allclose(ppl, np.full(4, 8.0, dtype=np.float32), label="perplexity")
    logprob = ctx.check("sequence_logprob", logits, tokens, mask, **TOL)
    pairs = np.array([4, 1, 5, 2], dtype=np.float32)
    check.allclose(logprob, -pairs * np.log(8.0), label="sequence_logprob", **TOL)


# Rejects log_of_softmax: probabilities as small as exp(-2e4) are not float32 numbers.
@case("extreme logits, spread over +-1e4: log-probabilities far below float32 underflow",
      public=True)
def extreme_logits(ctx):
    logits, tokens, mask = batch(ctx, [6, 3, 5, 2], "right", scale=3000.0)
    got = ctx.check("sequence_logprob", logits, tokens, mask, **TOL)
    check.finite(got, label="sequence_logprob")
    row = np.array([[[0.0, -200.0, -1e4, 1e4]] * 3], dtype=np.float32)
    tokens = np.array([[0, 1, 2]], dtype=np.int64)
    got = ctx.check("sequence_logprob", row, tokens, np.ones((1, 3), dtype=bool), **TOL)
    check.finite(got, label="sequence_logprob")


# Rejects multiplies_by_mask. Token 0 is banned (-inf everywhere) and is what the
# padding positions hold, so only uncounted pairs have a target of probability zero.
@case("a banned token (-inf logit) that appears only at padding positions")
def banned_padding_token(ctx):
    for side in ("right", "left"):
        logits, tokens, mask = batch(ctx, [3, 6, 2, 4], side)
        logits[:, :, 0] = -np.inf
        tokens = np.where(mask, np.maximum(tokens, 1), 0)
        for fn in BOTH:
            got = ctx.check(fn, logits, tokens, mask, **TOL)
            check.finite(got, label=fn)


@case("what the padding positions hold does not change the result")
def padding_content_is_ignored(ctx):
    for side in ("right", "left"):
        logits, tokens, mask = batch(ctx, [3, 5, 2, 4], side)
        other_logits = np.where(mask[:, :, None], logits, ctx.randn(4, 6, 8, scale=50.0))
        other_tokens = np.where(mask, tokens, ctx.randint(0, 8, (4, 6)))
        for fn in BOTH:
            first = ctx.check(fn, logits, tokens, mask, **TOL)
            second = ctx.check(fn, other_logits, other_tokens, mask, **TOL)
            check.allclose(second, first, label=f"{fn} after changing the padding", **TOL)


@case("rows with exactly two real tokens: one scored pair")
def shortest_rows(ctx):
    for side in ("right", "left"):
        check_both(ctx, *batch(ctx, [2, 2, 2, 2], side))
    check_both(ctx, *batch(ctx, [2, 2, 2, 2], "right", seq=2))


@case("a confident model: the target has almost all of the probability")
def confident_model(ctx):
    logits, tokens, mask = batch(ctx, [6, 4, 2, 5], "right")
    for b in range(4):
        for t in range(5):
            logits[b, t, tokens[b, t + 1]] += 12.0
    check_both(ctx, logits, tokens, mask)


# Square shapes: an axis mix-up between positions and vocabulary cannot hide.
@case("all axes the same length: B = T = V = 6")
def all_axes_equal(ctx):
    for side in ("right", "left"):
        check_both(ctx, *batch(ctx, [6, 3, 2, 5, 4, 2], side, seq=6, vocab=6))


@case("a batch of one, and a long row: T = 16")
def other_shapes(ctx):
    check_both(ctx, *batch(ctx, [4], "left"))
    check_both(ctx, *batch(ctx, [16, 9, 2, 12], "right", seq=16))
    check_both(ctx, *batch(ctx, [16, 9, 2, 12], "left", seq=16, scale=4.0))
