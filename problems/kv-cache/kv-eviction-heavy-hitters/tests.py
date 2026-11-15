"""Tests for kv-eviction-heavy-hitters.

`simulate` is the specification written as plain loops over batch rows and heads, in
numpy and independent of the oracle. Every case runs a decode (a prefill, then steps)
through the solution, feeding each returned cache and score tensor into the next call,
and compares every step with the simulation: the kept positions exactly, the scores to
float tolerance, the keys and values exactly (they are copies).

Kept positions are compared exactly, so the generated attention never leaves two
competing scores closer than GAP (a one-ulp difference between frameworks could
otherwise flip the order); cases about ties use dyadic weights, whose sums are exact.

JAX compiles every new shape: decodes start with a prefill of `budget` tokens, after
which every step has the same shapes.
"""

import numpy as np

from whetstone.judge import case, check

FN = "h2o_step"
B, H, D = 1, 2, 4  # the sizes most cases share
GAP = 1e-3
EXACT = {"rtol": 0.0, "atol": 0.0}


def simulate(scores, attn, budget, recent):
    """(scores, kept) of the statement for one step; scores may be None."""
    b, h, t, s = attn.shape
    received = attn.sum(axis=2, dtype=np.float32)
    if scores is None:
        total = received
    else:
        total = np.concatenate([scores, np.zeros((b, h, t), np.float32)], axis=2) + received
    n = min(s, budget)
    kept = np.zeros((b, h, n), dtype=np.int64)
    for i in range(b):
        for j in range(h):
            if s <= budget:
                kept[i, j] = np.arange(s)
                continue
            old = s - recent
            ranked = sorted(range(old), key=lambda p: (-total[i, j, p], p))
            kept[i, j] = sorted(ranked[: budget - recent]) + list(range(old, s))
    return total, kept


def min_gap(total, budget, recent):
    """The smallest difference between two scores that compete for a place."""
    s = total.shape[-1]
    if s <= budget or s - recent < 2:
        return np.inf
    competing = np.sort(total[:, :, : s - recent], axis=-1)
    return float(np.diff(competing, axis=-1).min())


def causal_attention(rng, b, h, t, s, sharp=2.0):
    """Random attention of t new queries over s entries: query i sees entries 0 .. s-t+i."""
    logits = rng.standard_normal((b, h, t, s)) * sharp
    visible = np.arange(s)[None, :] <= (s - t) + np.arange(t)[:, None]
    logits = np.where(visible, logits, -np.inf)
    weights = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return (weights / weights.sum(axis=-1, keepdims=True)).astype(np.float32)


def numbered(first, last, b=B, h=H, d=D):
    """Keys and values of tokens first .. last - 1: element 0 of a key is its token number."""
    n = last - first
    k = np.zeros((b, h, n, d), dtype=np.float32)
    k[..., 0] = np.arange(first, last)
    k[..., 1:] = (np.arange(first, last)[:, None] * 0.25 + np.arange(1, d)) % 3.0
    return k, 100.0 + k


def one_step(ctx, cache, scores, attn, budget, recent, what):
    """Call the solution once and compare everything with the simulation."""
    want_scores, want_kept = simulate(scores, attn, budget, recent)
    result = ctx.call(FN, cache, scores, attn, budget, recent)
    check.that(
        isinstance(result, tuple) and len(result) == 3,
        f"{FN} must return a tuple (cache, scores, kept)",
    )
    new_cache, new_scores, kept = result
    check.that(
        isinstance(new_cache, tuple) and len(new_cache) == 2,
        f"{FN}: the returned cache must be a tuple (k, v)",
    )
    check.that(
        np.issubdtype(np.asarray(kept).dtype, np.integer),
        f"{FN}: kept must be an integer tensor",
        f"got dtype {np.asarray(kept).dtype}",
    )
    check.equal(kept, want_kept, label=f"kept positions {what}")
    b, h = want_kept.shape[:2]
    rows, heads = np.arange(b)[:, None, None], np.arange(h)[None, :, None]
    check.allclose(new_scores, want_scores[rows, heads, want_kept], rtol=1e-4, atol=1e-5,
                   label=f"scores {what}")  # fmt: skip
    check.allclose(new_cache[0], cache[0][rows, heads, want_kept], **EXACT, label=f"keys {what}")
    check.allclose(new_cache[1], cache[1][rows, heads, want_kept], **EXACT,
                   label=f"values {what}")  # fmt: skip
    return new_cache, new_scores, kept


def decode(ctx, chunks, budget, recent, *, b=B, h=H, d=D, sharp=2.0, attention=None):
    """Append chunks of the given sizes one after another, evicting after each.

    The attention of each step is drawn here (or taken from `attention(step, t, s)`),
    over the entries the solution's own cache holds at that point.
    """
    cache, scores, first = None, None, 0
    for step, t in enumerate(chunks):
        k_new, v_new = numbered(first, first + t, b, h, d)
        first += t
        if cache is None:
            cache = (k_new, v_new)
        else:
            cache = (np.concatenate([cache[0], k_new], axis=2),
                     np.concatenate([cache[1], v_new], axis=2))  # fmt: skip
        s = cache[0].shape[2]
        if attention is not None:
            attn = attention(step, t, s)
        else:
            for _ in range(200):
                attn = causal_attention(ctx.rng, b, h, t, s, sharp)
                if min_gap(simulate(scores, attn, budget, recent)[0], budget, recent) > GAP:
                    break
            else:
                raise AssertionError("could not draw attention without a near-tie")
        what = f"at step {step + 1} ({t} new, {s} entries before eviction)"
        cache, scores, _ = one_step(ctx, cache, scores, attn, budget, recent, what)
    return cache, scores


def tokens_of(cache):
    """Token numbers a cache holds, per batch row and head (read from element 0 of the keys)."""
    return np.rint(cache[0][..., 0]).astype(int)


def rows_of(*rows):
    """Attention rows for one batch row and one head: shape (1, 1, T, S)."""
    return np.array(rows, dtype=np.float32)[None, None]


@case("the worked example: budget 4, recent 2, a prefill of 4 tokens and two steps", public=True)
def worked_example(ctx):
    k, v = numbered(0, 6, b=1, h=1, d=1)
    prefill = rows_of(
        [1.0, 0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0, 0.0],
        [0.25, 0.25, 0.5, 0.0],
        [0.5, 0.125, 0.125, 0.25],
    )
    cache, scores, kept = ctx.check(FN, (k[:, :, :4], v[:, :, :4]), None, prefill, 4, 2)
    check.equal(kept[0, 0], [0, 1, 2, 3], label="kept after the prefill")
    check.allclose(scores[0, 0], [2.25, 0.875, 0.625, 0.25], label="scores after the prefill")

    cache = (np.concatenate([cache[0], k[:, :, 4:5]], axis=2),
             np.concatenate([cache[1], v[:, :, 4:5]], axis=2))  # fmt: skip
    step = rows_of([0.125, 0.125, 0.5, 0.125, 0.125])
    cache, scores, kept = ctx.check(FN, cache, scores, step, 4, 2)
    check.equal(kept[0, 0], [0, 2, 3, 4], label="kept after step 1")
    check.allclose(scores[0, 0], [2.375, 1.125, 0.375, 0.125], label="scores after step 1")
    check.equal(tokens_of(cache)[0, 0], [0, 2, 3, 4], label="tokens cached after step 1")

    cache = (np.concatenate([cache[0], k[:, :, 5:6]], axis=2),
             np.concatenate([cache[1], v[:, :, 5:6]], axis=2))  # fmt: skip
    step = rows_of([0.25, 0.0, 0.5, 0.125, 0.125])
    cache, scores, kept = ctx.check(FN, cache, scores, step, 4, 2)
    check.equal(kept[0, 0], [0, 1, 3, 4], label="kept after step 2")
    check.allclose(scores[0, 0], [2.625, 1.125, 0.25, 0.125], label="scores after step 2")
    check.equal(tokens_of(cache)[0, 0], [0, 2, 4, 5], label="tokens cached after step 2")
    check.allclose(cache[1][0, 0, :, 0], [100.0, 102.0, 104.0, 105.0], label="values after step 2")


@case("under the budget nothing is evicted and the scores accumulate", public=True)
def under_budget(ctx):
    cache, scores = decode(ctx, [3, 1, 1, 1], budget=6, recent=2)
    check.equal(tokens_of(cache)[0, 0], list(range(6)), label="tokens cached")
    check.allclose(scores.sum(axis=-1), np.full((B, H), 6.0), rtol=1e-4,
                   label="sum of the scores (each query hands out 1)")  # fmt: skip


# Rejects latest_attention_only, scores_not_carried, and order_not_preserved: over ten
# evictions the accumulated score, not the last row, decides, and the survivors must
# stay aligned with their scores and in order.
@case("a decode of 10 steps at budget 6, recent 2: one eviction per step and head", public=True)
def long_decode(ctx):
    decode(ctx, [6] + [1] * 10, budget=6, recent=2)


# Rejects shared_across_heads: the two heads attend to different tokens.
@case("each head evicts its own tokens", public=True)
def heads_differ(ctx):
    def attention(step, t, s):
        attn = np.zeros((B, H, t, s), dtype=np.float32)
        if step == 0:  # head 0 dwells on the first token, head 1 on the later ones
            attn[:, 0] = [[1, 0, 0, 0], [0.75, 0.25, 0, 0], [0.75, 0, 0.25, 0], [1, 0, 0, 0]]
            attn[:, 1] = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0.5, 0.5, 0], [0, 0.25, 0.25, 0.5]]
        else:
            attn[:, 0, 0, [0, s - 1]] = 0.5
            attn[:, 1, 0, [s - 2, s - 1]] = 0.5
        return attn

    cache, _ = decode(ctx, [4, 1, 1], budget=4, recent=1, attention=attention)
    held = tokens_of(cache)
    check.that(
        not np.array_equal(held[0, 0], held[0, 1]),
        "both heads hold the same tokens, although they attended to different ones",
        f"head 0 holds tokens {held[0, 0].tolist()}, head 1 holds {held[0, 1].tolist()}",
    )


# Rejects no_recent_window: the newest tokens have received almost nothing yet.
@case("the newest tokens survive although their scores are the lowest", public=True)
def recent_survive(ctx):
    def attention(step, t, s):
        attn = causal_attention(np.random.default_rng(step), B, H, t, s, sharp=1.0)
        attn[..., s - 3 :] *= 0.01  # almost nothing for the three newest entries
        attn[..., 0] += 1.0 - attn.sum(axis=-1)
        return attn

    cache, _ = decode(ctx, [6, 1, 1, 1, 1], budget=6, recent=3, attention=attention)
    held = tokens_of(cache)
    check.equal(held[:, :, -3:], np.broadcast_to(np.array([7, 8, 9]), (B, H, 3)),
                label="the three newest tokens cached")  # fmt: skip


# Rejects ties_keep_newest. All weights are dyadic, so equal scores are exactly equal.
@case("equal scores: the older token is kept")
def ties_keep_older(ctx):
    def attention(step, t, s):
        attn = np.zeros((B, H, t, s), dtype=np.float32)
        if step == 0:
            attn[..., 0, 0] = 1.0
            attn[..., 1, :2] = 0.5
            attn[..., 2, :4] = [0.25, 0.25, 0.5, 0.0]
            attn[..., 3, :4] = [0.0, 0.0, 0.25, 0.75]
            attn[..., 4, :5] = [0.25, 0.25, 0.25, 0.0, 0.25]
            attn[..., 5, :6] = [0.0, 0.0, 0.0, 0.0, 0.5, 0.5]
        else:
            attn[..., 0, s - 1] = 1.0
        return attn

    # After the prefill: scores [2, 1, 1, .75, .75, .5]. Tokens 1 and 2 tie, 3 and 4 tie.
    cache, _ = decode(ctx, [6, 1, 1], budget=6, recent=2, attention=attention)
    check.equal(tokens_of(cache)[0, 0], [0, 1, 2, 3, 6, 7], label="tokens cached")
    cache, _ = decode(ctx, [6, 1], budget=3, recent=1, attention=attention)
    check.equal(tokens_of(cache)[0, 0], [0, 1, 6], label="tokens cached at budget 3")


# Rejects last_query_only: the score of a prefill is the column sum over all its queries.
@case("a first call longer than the budget: the prompt is cut down to the budget at once")
def long_prefill(ctx):
    decode(ctx, [9, 1, 1], budget=6, recent=2)


@case("several new tokens per step: column sums over the new queries, several evictions")
def multi_token_steps(ctx):
    decode(ctx, [6, 3, 3, 3], budget=6, recent=2)
    decode(ctx, [6, 3, 3], budget=6, recent=3)


@case("recent = 0 (scores alone decide) and recent = budget (a plain sliding window)")
def window_extremes(ctx):
    decode(ctx, [6, 1, 1, 1], budget=6, recent=0)
    cache, _ = decode(ctx, [6, 1, 1, 1], budget=6, recent=6)
    check.equal(tokens_of(cache)[0, 0], list(range(3, 9)), label="tokens cached with recent = 6")


@case("a budget of one heavy hitter: budget 3, recent 2")
def one_heavy_hitter(ctx):
    decode(ctx, [3, 1, 1, 1, 1, 1], budget=3, recent=2)


@case("random decodes: prefill sizes, step sizes, budgets, and windows")
def random_decodes(ctx):
    def prop(rng):
        budget, recent = (int(n) for n in rng.choice([(6, 2), (6, 4), (4, 1), (4, 2)]))
        tail = [int(n) for n in rng.choice([1, 1, 1, 2], size=5)]
        decode(ctx, [budget, *tail], budget=budget, recent=recent,
               sharp=float(rng.choice([0.5, 2.0, 4.0])))  # fmt: skip

    check.forall(prop, trials=6)


@case("other sizes: B = 2, H = 3, D = 8")
def other_sizes(ctx):
    decode(ctx, [6, 1, 1, 1, 1], budget=6, recent=2, b=2, h=3, d=8)
