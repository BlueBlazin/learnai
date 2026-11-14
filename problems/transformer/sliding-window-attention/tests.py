"""Tests for sliding-window-attention.

The mask is compared exactly, and attention numerically, with the oracle. Property
cases state the two consequences in the statement without the oracle: a window at
least as long as the sequence gives causal attention, and keys outside a query's window
have no influence on its output.
"""

import numpy as np

from whetstone.judge import case, check

MASK = "window_mask"
ATTN = "sliding_window_attention"
TOL = {"rtol": 1e-4, "atol": 1e-5}  # the tolerance stated in Constraints


def qkv(ctx, b, h, t, d, scale=1.0):
    return (
        ctx.randn(b, h, t, d, scale=scale),
        ctx.randn(b, h, t, d, scale=scale),
        ctx.randn(b, h, t, d),
    )


def causal_attention(q, k, v):
    """Plain causal attention in numpy (float64), for the causal-limit property."""
    q, k, v = (a.astype(np.float64) for a in (q, k, v))
    seq = q.shape[-2]
    scores = q @ np.swapaxes(k, -2, -1) / np.sqrt(q.shape[-1])
    scores = np.where(np.tril(np.ones((seq, seq), dtype=bool)), scores, -np.inf)
    weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
    weights /= weights.sum(axis=-1, keepdims=True)
    return (weights @ v).astype(np.float32)


@case("the worked example: window_mask(4, 2), and three tokens with window 2", public=True)
def worked_example(ctx):
    mask = ctx.check(MASK, 4, 2)
    expected_mask = np.array([[1, 0, 0, 0], [1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]], dtype=bool)
    check.equal(mask, expected_mask, label=MASK)
    x = np.array([[[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]]], dtype=np.float32)
    got = ctx.check(ATTN, x, x, x, 2, **TOL)
    expected = np.array([[[[1.0, 0.0], [0.330238, 0.669762], [0.669762, 1.0]]]], dtype=np.float32)
    check.allclose(got, expected, atol=1e-5, label=ATTN)


# Rejects window_plus_one, symmetric_window, window_from_sequence_end, and inverted_mask
# on the mask alone, where the failure shows the two matrices side by side.
@case("masks for several sizes: window 1, window inside the sequence, window >= T", public=True)
def masks(ctx):
    for seq, window in [(5, 1), (5, 3), (6, 2), (7, 6), (4, 4), (4, 9), (1, 1), (1, 3)]:
        ctx.check(MASK, seq, window)


@case("a random batch: shape (2, 3, 6, 8), window 3", public=True)
def random_batch(ctx):
    ctx.check(ATTN, *qkv(ctx, 2, 3, 6, 8), 3, **TOL)
    ctx.check(ATTN, *qkv(ctx, 2, 3, 6, 8), window=2, **TOL)


@case("with window >= T the result is ordinary causal attention", public=True)
def causal_limit(ctx):
    def prop(rng):
        q, k, v = qkv(ctx, 2, 3, 6, 8)
        window = 6 + int(rng.integers(0, 4))
        got = ctx.call(ATTN, q, k, v, window)
        check.shape(got, (2, 3, 6, 8), label=ATTN)
        check.allclose(
            got,
            causal_attention(q, k, v),
            atol=1e-5,
            label=f"attention with T = 6 and window = {window} versus causal attention",
        )

    check.forall(prop, trials=4)


# Rejects window_plus_one (the key at i - window leaks in) and mask_after_softmax (the
# hidden scores still sit in the denominator).
@case("keys and values older than the window do not influence a query's output", public=True)
def outside_window_has_no_influence(ctx):
    def prop(rng):
        seq = 8
        window = int(rng.integers(1, 5))
        i = int(rng.integers(window, seq))  # a query with at least one key behind its window
        q, k, v = qkv(ctx, 2, 3, seq, 8)
        cut = i - window + 1  # the first key inside the window of query i
        k2, v2 = k.copy(), v.copy()
        k2[:, :, :cut] = ctx.randn(2, 3, cut, 8, scale=3.0)
        v2[:, :, :cut] = ctx.randn(2, 3, cut, 8, scale=3.0)
        before = ctx.call(ATTN, q, k, v, window)
        after = ctx.call(ATTN, q, k2, v2, window)
        check.shape(before, (2, 3, seq, 8), label=ATTN)
        check.finite(before, label=ATTN)
        check.allclose(
            after[:, :, i],
            before[:, :, i],
            atol=1e-5,
            label=f"output of query {i} with window {window} after changing keys 0 to {cut - 1}",
        )
        # The oldest key inside the window does count: moving its value moves the output.
        v3 = v.copy()
        v3[:, :, cut] += 100.0
        moved = np.abs(ctx.call(ATTN, q, k, v3, window)[:, :, i] - before[:, :, i]).max()
        check.that(
            bool(moved > 1e-2),
            f"query {i} with window {window} ignores key {cut}, the oldest key inside its window",
            f"changing the value at position {cut} by 100 moved the output by {moved:.3g}",
        )

    check.forall(prop, trials=5)


# Rejects symmetric_window with a message that names the cause.
@case("changing later tokens never changes earlier outputs")
def no_future_leak(ctx):
    def prop(rng):
        seq = 8
        window = int(rng.integers(2, 6))
        t = int(rng.integers(1, seq))
        q, k, v = qkv(ctx, 2, 3, seq, 8)
        k2, v2 = k.copy(), v.copy()
        k2[:, :, t:] = ctx.randn(2, 3, seq - t, 8, scale=3.0)
        v2[:, :, t:] = ctx.randn(2, 3, seq - t, 8, scale=3.0)
        before = ctx.call(ATTN, q, k, v, window)
        after = ctx.call(ATTN, q, k2, v2, window)
        check.allclose(
            after[:, :, :t],
            before[:, :, :t],
            atol=1e-5,
            label=f"outputs before position {t} after changing keys from position {t} on",
        )

    check.forall(prop, trials=4)


@case("window 1: every token attends only to itself, so the output is v")
def window_one(ctx):
    q, k, v = qkv(ctx, 2, 3, 6, 8)
    got = ctx.check(ATTN, q, k, v, 1, **TOL)
    check.allclose(got, v, atol=1e-5, label="attention with window 1")


# The three windows around the sequence length: T - 1 hides exactly one key (key 0 from
# the last query), T and T + 1 hide nothing.
@case("windows at the sequence length: T - 1, T, and T + 1")
def window_at_sequence_length(ctx):
    q, k, v = qkv(ctx, 2, 3, 6, 8)
    for window in (5, 6, 7):
        ctx.check(ATTN, q, k, v, window, **TOL)


@case("a long sequence with a short window: T = 24, window 4, and a single token")
def long_sequence(ctx):
    ctx.check(MASK, 24, 4)
    ctx.check(ATTN, *qkv(ctx, 1, 2, 24, 8), 4, **TOL)
    ctx.check(ATTN, *qkv(ctx, 2, 3, 1, 8), 4, **TOL)


# B = H = T = d: every axis has the same length, which hides transposed axes.
@case("all axes the same length: B = H = T = d = 4, windows 2 and 3")
def all_axes_equal(ctx):
    ctx.check(ATTN, *qkv(ctx, 4, 4, 4, 4), 2, **TOL)
    ctx.check(ATTN, *qkv(ctx, 4, 4, 4, 4), 3, **TOL)


@case("large scores: nearly one-hot attention inside the window")
def sharp_attention(ctx):
    got = ctx.check(ATTN, *qkv(ctx, 2, 3, 6, 8, scale=4.0), 3, **TOL)
    check.finite(got, label=ATTN)


@case("random sizes and windows")
def random_sizes(ctx):
    for _ in range(2):
        b, h, d = (int(n) for n in ctx.rng.integers(1, 6, size=3))
        seq = int(ctx.rng.integers(3, 13))
        window = int(ctx.rng.integers(1, seq))  # always shorter than the sequence
        ctx.check(MASK, seq, window)
        ctx.check(ATTN, *qkv(ctx, b, h, seq, d), window, **TOL)
