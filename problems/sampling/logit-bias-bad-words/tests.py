"""Tests for logit-bias-bad-words. Every case compares the solution with the oracle.

The result is the input logits plus a bias, with some entries replaced by -inf, so the
comparison is exact up to one float32 addition and there is nothing to keep clear of.
Token ids, the bias, the allow-list, the banned sequences, and the history are plain
Python values, as a server would hold them for a request. All cases but the last use a
vocabulary of 12 tokens, and histories over a few of them so that sequences do match.
"""

import numpy as np

from whetstone.judge import case, check

FN = "apply_bias"
INF = np.inf
V = 12


def f32(values):
    return np.array(values, dtype=np.float32)


def run(ctx, logits, bias=None, allowed=None, bad_words=(), history=()):
    """Compare with the oracle; returns the solution's result."""
    bias = dict(bias or {})
    allowed = None if allowed is None else [int(t) for t in allowed]
    bad_words = [[int(t) for t in word] for word in bad_words]
    history = [int(t) for t in history]
    return ctx.check(FN, f32(logits), bias, allowed, bad_words, history)


def removed(out):
    return [int(i) for i in np.flatnonzero(np.isneginf(out))]


@case(
    "the worked example: a bias, an allow-list, and five banned sequences, two matching",
    public=True,
)
def worked_example(ctx):
    logits = f32([1.0, 2.0, 0.5, -1.0, 3.0, 0.0, 1.5, -0.5])
    got = ctx.call(
        FN,
        logits,
        {1: -1.5, 6: 2.0, 7: 100.0},
        [0, 1, 2, 3, 4, 6],
        [[4], [5, 2, 3], [2, 6], [2, 5, 0], [3, 2, 1]],
        [3, 5, 2],
    )
    expected = f32([1.0, 0.5, 0.5, -INF, -INF, -INF, -INF, -INF])
    check.allclose(got, expected, label=FN, dtype=True)


@case("the bias alone: added to the logit, also for large values and masked logits", public=True)
def bias_only(ctx):
    # Catches a bias that is assigned instead of added.
    logits = ctx.randn(V, scale=3.0)
    logits[4] = logits[9] = -INF
    run(ctx, logits)
    run(ctx, logits, {0: 1.25, 3: -2.5, 4: 50.0, 7: 100.0, 9: -100.0, 11: -0.75})
    run(ctx, logits, {5: 0.0}, history=[1, 2, 3])


@case("the allow-list: every other token is removed, whatever its bias", public=True)
def allow_list(ctx):
    # Catches a bias that is applied after the mask and makes a removed token finite.
    logits = ctx.randn(V, scale=3.0)
    bias = {0: 100.0, 2: -3.0, 5: 100.0, 6: 4.0, 10: 100.0}
    run(ctx, logits, bias, allowed=[2, 6, 7])
    run(ctx, logits, bias, allowed=[5, 5, 0, 11, 5, 0])  # duplicates
    run(ctx, logits, bias, allowed=[3])
    run(ctx, logits, bias, allowed=list(range(V)))


@case(
    "a banned sequence removes its last token only while the history ends with the rest",
    public=True,
)
def generation_walk(ctx):
    # The same request at every step of a growing history, as a server would see it.
    # [1, 2, 3] bans token 3 after "... 1 2" only; [2, 2] bans token 2 after a 2.
    logits = ctx.randn(V, scale=2.0)
    bad_words = [[1, 2, 3], [2, 2], [0, 1, 2, 4]]
    stream = [0, 1, 2, 5, 1, 2, 2, 1, 0, 1, 2, 3, 1, 2]
    expected = {3: [2, 3, 4], 6: [2, 3], 7: [2], 11: [2, 3, 4], 14: [2, 3]}
    for step in range(len(stream) + 1):
        out = run(ctx, logits, bad_words=bad_words, history=stream[:step])
        check.equal(removed(out), expected.get(step, []), label=f"removed tokens at step {step}")


@case("one-token sequences are removed at every step, with an empty history too")
def single_token_words(ctx):
    # Catches history[-0:], which is the whole history and equals [] only when it is empty.
    logits = ctx.randn(V, scale=2.0)
    bad_words = [[3], [8], [3], [5, 8]]
    for history in ([], [3], [1, 4, 6, 6, 2], [8, 3, 8, 5]):
        run(ctx, logits, bad_words=bad_words, history=history)
    run(ctx, logits, {3: 100.0, 8: -1.0}, [3, 4, 8], bad_words, [7])


@case("the prefix must match in order: the same tokens reversed or rotated do not count")
def order_matters(ctx):
    logits = ctx.randn(V, scale=2.0)
    bad_words = [[4, 5, 6, 0], [9, 2, 1], [7, 7, 3, 2]]
    for history in (
        [6, 5, 4],
        [5, 6, 4],
        [4, 5, 6],
        [1, 2, 9],
        [2, 9],
        [9, 2],
        [7, 3, 7],
        [3, 7, 7],
        [7, 7, 3],
    ):
        run(ctx, logits, bad_words=bad_words, history=history)


@case("the prefix must be at the end: earlier in the history, or cut off, it does not count")
def end_of_history(ctx):
    logits = ctx.randn(V, scale=2.0)
    bad_words = [[1, 2, 3], [6, 0, 4, 5]]
    for history in (
        [1, 2, 7],  # the prefix, then another token
        [1, 2, 7, 7, 7, 8],
        [7, 1, 2],
        [2],  # shorter than the prefix
        [1],
        [0, 4],
        [6, 0, 4],  # the history is exactly the prefix
        [6, 0, 4, 9],
        [1, 2, 6, 0, 4, 1, 2],
    ):
        run(ctx, logits, bad_words=bad_words, history=history)


@case("the tokens before the last one of a matching sequence are not removed")
def only_the_last_token(ctx):
    logits = ctx.randn(V, scale=2.0)
    out = run(ctx, logits, bad_words=[[5, 9, 2, 10]], history=[0, 5, 9, 2])
    check.equal(removed(out), [10], label="removed tokens")
    out = run(ctx, logits, {5: 1.0, 9: -1.0}, bad_words=[[5, 9], [9, 5, 9, 1]], history=[9, 5])
    check.equal(removed(out), [9], label="removed tokens")


@case("an allowed token that completes a banned sequence is removed")
def allow_list_and_bad_words(ctx):
    logits = ctx.randn(V, scale=2.0)
    bias = {4: 100.0, 6: 2.0, 1: -2.0}
    out = run(ctx, logits, bias, [1, 4, 6, 8], [[2, 4], [3, 6], [8]], [0, 2])
    check.equal(removed(out), [0, 2, 3, 4, 5, 7, 8, 9, 10, 11], label="removed tokens")
    run(ctx, logits, bias, [1, 4, 6, 8], [[2, 4], [3, 6]], [2, 3])
    run(ctx, logits, bias, [4], [[4]], [])  # nothing is left: all -inf


@case("overlapping sequences: several match at once, share a last token, repeat a token")
def overlapping_sequences(ctx):
    logits = ctx.randn(V, scale=2.0)
    bad_words = [[7, 7, 7], [7, 3], [1, 7, 3], [7, 7, 0], [2, 7, 7, 5], [7, 7, 7, 7, 6]]
    for history in ([7], [7, 7], [1, 7], [2, 7, 7], [7, 7, 7, 7], [7, 7, 7, 7, 7], [7, 3]):
        run(ctx, logits, {3: 5.0, 7: -5.0}, bad_words=bad_words, history=history)


@case("random requests over a small alphabet, where sequences match often")
def random_requests(ctx):
    rng = ctx.rng
    for _ in range(40):
        logits = ctx.randn(V, scale=3.0)
        logits[rng.random(V) < 0.1] = -INF
        bias = {
            int(t): float(np.round(rng.uniform(-100, 100), 2))
            for t in rng.choice(V, size=int(rng.integers(0, 6)), replace=False)
        }
        allowed = None
        if rng.random() < 0.5:
            allowed = rng.integers(0, V, size=int(rng.integers(1, 10))).tolist()
        bad_words = [
            rng.integers(0, 4, size=int(rng.integers(1, 5))).tolist()
            for _ in range(int(rng.integers(0, 7)))
        ]
        history = rng.integers(0, 4, size=int(rng.integers(0, 9))).tolist()
        run(ctx, logits, bias, allowed, bad_words, history)


@case("a vocabulary of 5000 tokens, a history of 2000, and 64 banned sequences")
def large_request(ctx):
    rng = ctx.rng
    vocab = 5000
    logits = ctx.randn(vocab, scale=3.0)
    history = rng.integers(0, vocab, size=2000).tolist()
    bad_words = [rng.integers(0, vocab, size=int(rng.integers(1, 9))).tolist() for _ in range(48)]
    # Sixteen sequences that do match: the last 0 to 7 tokens of the history plus a token.
    bad_words += [history[len(history) - n % 8 :] + [100 + n] for n in range(16)]
    bias = {int(t): float(rng.uniform(-100, 100)) for t in rng.choice(vocab, 300, replace=False)}
    run(ctx, logits, bias, None, bad_words, history)
    run(ctx, logits, bias, rng.choice(vocab, 1000, replace=False).tolist(), bad_words, history)
