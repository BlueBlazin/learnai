"""Tests for snapkv-compress.

`select` is the specification written as loops over batch rows, heads, and positions,
in numpy and independent of the oracle. Positions are compared exactly. Max pooling
makes exact ties the normal case (neighbors share a maximum), and the statement's tie
rule decides them; what must not happen is two *different* votes so close that float
rounding orders them differently in different frameworks, so `prompt_attention` redraws
until distinct votes of a head differ by at least GAP.

JAX compiles every new shape: most cases use prompts of 24 tokens.
"""

import numpy as np

from learnai.judge import case, check

FN = "snapkv_select"
B, H, L = 1, 2, 24
GAP = 1e-4


def votes(attn, window):
    """What each prefix position receives from the window's queries: (B, H, L - window)."""
    prefix = attn.shape[-1] - window
    return attn[:, :, prefix:, :prefix].astype(np.float64).sum(axis=2)


def select(attn, window, budget, kernel):
    b, h, length, _ = attn.shape
    if length <= budget:
        return np.broadcast_to(np.arange(length), (b, h, length)).copy()
    prefix = length - window
    vote = votes(attn, window)
    out = np.zeros((b, h, budget), dtype=np.int64)
    for i in range(b):
        for j in range(h):
            pooled = [
                vote[i, j, max(0, p - kernel // 2) : p + kernel // 2 + 1].max()
                for p in range(prefix)
            ]
            ranked = sorted(range(prefix), key=lambda p: (-pooled[p], p))
            out[i, j] = sorted(ranked[: budget - window]) + list(range(prefix, length))
    return out


def min_vote_gap(attn, window):
    """The smallest difference between two distinct votes of one head."""
    ordered = np.sort(votes(attn, window), axis=-1)
    gaps = np.diff(ordered, axis=-1)
    return float(gaps[gaps > 0].min()) if (gaps > 0).any() else np.inf


def prompt_attention(rng, b, h, length, window, *, sharp=2.5, recency=0.0):
    """Causal attention of a prompt: softmax of random scores, optionally biased to recent."""
    for _ in range(200):
        scores = rng.standard_normal((b, h, length, length)) * sharp
        scores += recency * np.arange(length) / length
        scores = np.where(np.tril(np.ones((length, length), dtype=bool)), scores, -np.inf)
        weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
        attn = (weights / weights.sum(axis=-1, keepdims=True)).astype(np.float32)
        if min_vote_gap(attn, window) >= GAP:
            return attn
    raise AssertionError("could not draw attention without near-ties")


def check_select(ctx, attn, window, budget, kernel, what="kept positions"):
    got = ctx.call(FN, attn, window, budget, kernel)
    check.that(
        np.issubdtype(np.asarray(got).dtype, np.integer),
        f"{FN} must return an integer tensor",
        f"got dtype {np.asarray(got).dtype}",
    )
    check.equal(got, select(attn, window, budget, kernel), label=what)
    return got


def example_attention():
    attn = np.zeros((1, 1, 10, 10), dtype=np.float32)
    attn[0, 0, :8, 0] = 1.0  # the queries before the window look at position 0 only
    attn[0, 0, 8] = [0.02, 0.02, 0.40, 0.02, 0.02, 0.02, 0.20, 0.02, 0.28, 0.0]
    attn[0, 0, 9] = [0.02, 0.02, 0.30, 0.02, 0.02, 0.02, 0.10, 0.06, 0.14, 0.30]
    return attn


@case("the worked example: 10 tokens, window 2, budget 5, kernel 3", public=True)
def worked_example(ctx):
    attn = example_attention()
    got = ctx.check(FN, attn, 2, 5, 3)
    check.equal(got[0, 0], [1, 2, 3, 8, 9], label="kept positions")
    # The same votes without pooling: the three highest votes, not a cluster.
    got = ctx.check(FN, attn, 2, 5, 1)
    check.equal(got[0, 0], [2, 6, 7, 8, 9], label="kept positions with kernel 1")
    # One more slot: positions 5, 6, and 7 tie at 0.30, and the lowest is taken.
    got = ctx.check(FN, attn, 2, 6, 3)
    check.equal(got[0, 0], [1, 2, 3, 5, 8, 9], label="kept positions with budget 6")


@case("a prompt that fits the budget is kept whole", public=True)
def fits_budget(ctx):
    attn = prompt_attention(ctx.rng, B, H, 8, 4)
    for budget in (8, 12):
        got = check_select(ctx, attn, 4, budget, 5)
        check.equal(got, np.broadcast_to(np.arange(8), (B, H, 8)), label="kept positions")


# Rejects shared_across_heads (and anything that votes over the head axis).
@case("each head selects its own positions", public=True)
def per_head(ctx):
    attn = np.zeros((B, H, L, L), dtype=np.float32)
    attn[:, :, np.arange(L), np.arange(L)] = 1.0  # every query looks at itself ...
    attn[:, 0, 20:, :] = 0.0
    attn[:, 0, 20:, 3] = [0.5, 0.5, 0.25, 0.75]  # ... except the window: head 0 at 3 and 15,
    attn[:, 0, 20:, 15] = [0.5, 0.5, 0.75, 0.25]
    attn[:, 1, 20:, :] = 0.0
    attn[:, 1, 20:, 9] = 1.0  # head 1 at position 9
    got = check_select(ctx, attn, 4, 10, 3)
    check.equal(got[0, 0], [2, 3, 4, 14, 15, 16, 20, 21, 22, 23], label="positions of head 0")
    check.equal(got[0, 1], [0, 1, 2, 8, 9, 10, 20, 21, 22, 23], label="positions of head 1")


# Rejects pools_across_boundary, window_columns_compete, and all_rows_vote: with a
# recency bias the window's own columns and the rows before it carry the largest weights.
@case("only the window's rows vote, and only for positions before the window", public=True)
def window_rows_prefix_columns(ctx):
    # The window's queries put half their weight on the window's first position (20).
    attn = np.zeros((B, H, L, L), dtype=np.float32)
    attn[:, :, np.arange(L), np.arange(L)] = 1.0
    attn[:, :, 20:, :] = 0.0
    attn[:, :, 20:, 20] = 0.5
    attn[:, :, 20:, 5] = 0.3125
    attn[:, :, 20:, 12] = 0.1875
    got = check_select(ctx, attn, 4, 10, 5)
    check.equal(got[0, 0], [3, 4, 5, 6, 7, 10, 20, 21, 22, 23], label="kept positions")
    attn = prompt_attention(ctx.rng, B, H, L, 4, recency=6.0)
    check_select(ctx, attn, 4, 10, 5)
    attn = prompt_attention(ctx.rng, B, H, L, 6, recency=6.0)
    check_select(ctx, attn, 6, 12, 3)


# Rejects no_pooling, average_pooling, and kernel_as_radius.
@case("random prompts of 24 tokens: kernels 1, 3, 5, and 7")
def kernels(ctx):
    for kernel in (1, 3, 5, 7):
        attn = prompt_attention(ctx.rng, B, H, L, 4)
        check_select(ctx, attn, 4, 10, kernel, what=f"kept positions with kernel {kernel}")


# Rejects ties_to_highest and unsorted: one peak, a plateau of equal pooled votes around
# it, and a budget that cuts the plateau.
@case("a plateau cut by the budget: equal pooled votes are taken from the lowest position")
def plateau_cut(ctx):
    attn = np.zeros((B, H, L, L), dtype=np.float32)
    attn[:, :, np.arange(L), np.arange(L)] = 1.0
    attn[:, :, 20:, :] = 0.0
    attn[:, 0, 20:, 10] = 1.0  # head 0: one peak at 10, pooled plateau 8 .. 12
    attn[:, 1, 20:, 0] = 0.5  # head 1: peaks at both edges of the prefix
    attn[:, 1, 20:, 19] = 0.5
    got = check_select(ctx, attn, 4, 7, 5)
    check.equal(got[0, 0], [8, 9, 10, 20, 21, 22, 23], label="positions of head 0")
    check.equal(got[0, 1], [0, 1, 2, 20, 21, 22, 23], label="positions of head 1")
    got = check_select(ctx, attn, 4, 12, 5)
    check.equal(got[0, 0], [0, 1, 2, 8, 9, 10, 11, 12, 20, 21, 22, 23],
                label="positions of head 0 at budget 12")  # fmt: skip


@case("the budget equals the window (nothing but the window) and exceeds it by one")
def budget_extremes(ctx):
    attn = prompt_attention(ctx.rng, B, H, L, 4)
    got = check_select(ctx, attn, 4, 4, 5)
    check.equal(got, np.broadcast_to(np.arange(20, 24), (B, H, 4)), label="kept positions")
    check_select(ctx, attn, 4, 5, 5)
    check_select(ctx, attn, 4, 23, 5)


@case("a window of one query, and a kernel wider than the prefix")
def small_window_wide_kernel(ctx):
    attn = prompt_attention(ctx.rng, B, H, L, 1)
    check_select(ctx, attn, 1, 10, 3)
    attn = prompt_attention(ctx.rng, B, H, 8, 4)
    check_select(ctx, attn, 4, 6, 9)


@case("sharp and flat attention, several budgets")
def sharp_and_flat(ctx):
    for sharp, budget in ((0.3, 10), (6.0, 10), (2.5, 16), (2.5, 6)):
        attn = prompt_attention(ctx.rng, B, H, L, 4, sharp=sharp)
        check_select(ctx, attn, 4, budget, 5, what=f"kept positions (budget {budget})")


@case("other sizes: B = 2, H = 3, 16 tokens")
def other_sizes(ctx):
    attn = prompt_attention(ctx.rng, 2, 3, 16, 4)
    check_select(ctx, attn, 4, 9, 3)
