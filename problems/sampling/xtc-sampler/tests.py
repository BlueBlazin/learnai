"""Tests for xtc-sampler.

Numeric cases compare the solution with the oracle. Three comparisons decide the
output, and each is kept clear of float32 rounding (checked in float64 by `clear`):

- a probability against the threshold: at least P_MARGIN apart, except for the exact
  case of two equal logits in an otherwise masked row (probability exactly 0.5);
- two top choices against each other: equal logits, or at least X_MARGIN apart;
- u against probability: at least U_MARGIN apart (`coins` draws u that way).

The statistical cases check how often XTC acts: with independent uniforms, the top
token of a row with several top choices is removed with frequency `probability`. Two
checks with the default delta of 1e-10 keep the false-failure rate below 1e-9. They
use probabilities that float32 represents exactly, so `u < probability` is the same
comparison in every framework.
"""

import numpy as np

from whetstone.judge import case, check

FN = "xtc_filter"
INF = np.inf
P_MARGIN = 1e-4
X_MARGIN = 1e-3
U_MARGIN = 1e-4
N = 20_000


def f32(values):
    return np.array(values, dtype=np.float32)


def softmax(logits):
    x = np.asarray(logits, dtype=np.float64)
    e = np.exp(x - np.where(np.isfinite(x), x, -INF).max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def clear(logits, threshold, exact=False):
    """Whether no comparison of these rows can be decided by float32 rounding."""
    probs = softmax(logits)
    if not exact and np.abs(probs - threshold).min() < P_MARGIN:
        return False
    for row, p in zip(np.asarray(logits, dtype=np.float64), probs, strict=True):
        top = np.unique(row[p >= threshold])
        if len(top) > 1 and np.diff(top).min() < X_MARGIN:
            return False
    return True


def pick_threshold(ctx, logits, low=0.02, high=0.3):
    while True:
        threshold = float(f"{ctx.rng.uniform(low, high):.3g}")
        if clear(logits, threshold):
            return threshold


def coins(ctx, rows, probability):
    """One uniform per row in [0, 1), at least U_MARGIN away from `probability`."""
    u = ctx.rng.random(rows, dtype=np.float32)
    while True:
        close = np.abs(u.astype(np.float64) - probability) < U_MARGIN
        if not close.any():
            return u
        u[close] = ctx.rng.random(int(close.sum()), dtype=np.float32)


def check_fixed(ctx, logits, threshold, probability, u, exact=False):
    logits, u = f32(logits), f32(u)
    if not clear(logits, threshold, exact):  # an authoring error, not a solution failure
        raise RuntimeError("ambiguous test input: a comparison is within its margin")
    if np.abs(u.astype(np.float64) - probability).min() < U_MARGIN and 0 < probability < 1:
        raise RuntimeError("ambiguous test input: u is within U_MARGIN of probability")
    return ctx.check(FN, logits, threshold, probability, u)


def log_probs(rows):
    with np.errstate(divide="ignore"):
        return np.log(f32(rows))


def peaked(ctx, rows, vocab, scale=2.0):
    """Random rows with a few leading tokens, so that most rows have several top choices."""
    return ctx.randn(rows, vocab, scale=scale)


def acts_at_rate(ctx, row, threshold, probability):
    logits = np.tile(row, (N, 1))
    u = ctx.rng.random(N, dtype=np.float32)
    got = np.asarray(ctx.call(FN, logits, threshold, probability, u))
    check.shape(got, logits.shape, label=FN)
    acted = ctx.expected(FN, row[None], threshold, 1.0, f32([0.0]))[0]
    is_acted = (got == acted).all(axis=1)
    is_plain = (got == row).all(axis=1)
    check.that(
        bool((is_acted | is_plain).all()),
        f"{FN}: a row is neither unchanged nor the row with its top choices removed",
        f"logits = {row}\nthreshold = {threshold}\nfirst such row = {got[~(is_acted | is_plain)][:1]}",
    )
    top_removed = np.isneginf(got[:, int(np.argmax(row))]).astype(np.float64)
    check.mean(top_removed, probability, low=0.0, high=1.0, label="rows with the top token removed")


@case(
    "the worked example: a row that is filtered, the same row with a larger u, a confident row",
    public=True,
)
def worked_example(ctx):
    logits = log_probs(
        [
            [0.45, 0.25, 0.15, 0.08, 0.07],
            [0.45, 0.25, 0.15, 0.08, 0.07],
            [0.92, 0.05, 0.02, 0.01, 0.0],
        ]
    )
    got = ctx.call(FN, logits, 0.1, 0.5, f32([0.2, 0.8, 0.3]))
    remove = np.zeros(logits.shape, dtype=bool)
    remove[0, :2] = True
    check.allclose(got, np.where(remove, -INF, logits).astype(np.float32), label=FN, dtype=True)


@case("a random batch: every row has its own u, and its own number of top choices", public=True)
def random_batch(ctx):
    # Catches one coin for the whole batch, an inverted coin, and the wrong survivor.
    logits = peaked(ctx, 6, 10)
    for probability in (0.3, 0.5, 0.8):
        threshold = pick_threshold(ctx, logits)
        check_fixed(ctx, logits, threshold, probability, coins(ctx, 6, probability))


@case("a single token above the threshold is never removed; a threshold above 0.5", public=True)
def single_top_choice(ctx):
    # Every row is "filtered" (u = 0, probability = 1) and must come back unchanged.
    logits = log_probs(
        [
            [0.9, 0.04, 0.03, 0.02, 0.01, 0.0],
            [0.02, 0.03, 0.85, 0.05, 0.03, 0.02],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.3, 0.14, 0.14, 0.14, 0.14, 0.14],
        ]
    )
    u = f32([0.0, 0.0, 0.0, 0.0])
    got = check_fixed(ctx, logits, 0.2, 1.0, u)
    check.allclose(got, logits, label="rows with one top choice")
    wide = log_probs([[0.6, 0.3, 0.1, 0.0, 0.0, 0.0], [0.4, 0.3, 0.2, 0.1, 0.0, 0.0]])
    for threshold in (0.55, 0.7, 1.0):
        got = check_fixed(ctx, wide, threshold, 1.0, u[:2])
        check.allclose(got, wide, label="rows under a threshold above 0.5")


@case(
    "how often XTC acts: 20,000 rows, probability 0.25, the top token is removed in a quarter",
    public=True,
)
def acts_with_probability(ctx):
    acts_at_rate(ctx, log_probs([0.4, 0.3, 0.15, 0.1, 0.05, 0.0]), 0.12, 0.25)


@case("probability 0 never acts and probability 1 always does, whatever u is")
def probability_extremes(ctx):
    logits = peaked(ctx, 6, 10)
    threshold = pick_threshold(ctx, logits)
    u = f32([0.0, 0.999999, 0.5, 0.0, 0.25, 0.75])
    got = check_fixed(ctx, logits, threshold, 0.0, u)
    check.allclose(got, logits, label="rows with probability 0")
    check_fixed(ctx, logits, threshold, 1.0, u)


@case("top choices tied for least likely: the highest index survives; a 0.5 / 0.5 row")
def tied_survivors(ctx):
    # Equal logits are equal in every framework. Catches keeping every tied token
    # (mlx-lm's rule) and keeping the lowest index.
    logits = f32(
        [
            [3.0, 1.0, 2.0, 1.0, -4.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [2.0, -5.0, 2.0, -5.0, 2.0, -6.0],
            [1.0, 1.0, -INF, 3.0, -INF, -7.0],
        ]
    )
    u = f32([0.0, 0.1, 0.2, 0.3])
    for threshold in (0.02, 0.05, 0.12):
        check_fixed(ctx, logits, threshold, 0.5, u)
    # Two tokens at exactly 0.5 with a threshold of 0.5: both are top choices.
    halves = f32([[-INF, 1.5, -INF, 1.5, -INF, -INF], [7.0, 7.0, -INF, -INF, -INF, -INF]])
    check_fixed(ctx, halves, 0.5, 1.0, u[:2], exact=True)


@case("masked logits (-inf) stay masked and are never top choices")
def masked_logits(ctx):
    logits = peaked(ctx, 6, 10)
    logits[0, 2] = logits[0, 5] = logits[2, 0] = -INF
    logits[1, 2:] = -INF
    logits[3, :7] = -INF
    logits[4, 1:] = -INF
    for probability in (0.5, 1.0):
        threshold = pick_threshold(ctx, logits, 0.02, 0.4)
        check_fixed(ctx, logits, threshold, probability, coins(ctx, 6, probability))


@case("tokens below the threshold are untouched, also when they are the majority")
def tail_untouched(ctx):
    # Row 0 keeps its 0.11 token and all five tail tokens; nothing is renormalized.
    logits = log_probs(
        [
            [0.35, 0.3, 0.11, 0.06, 0.05, 0.05, 0.04, 0.04, 0.0, 0.0],
            [0.05, 0.05, 0.3, 0.05, 0.3, 0.05, 0.15, 0.03, 0.01, 0.01],
        ]
    ) + f32([[250.0], [-600.0]])
    for threshold in (0.1, 0.2):
        check_fixed(ctx, logits, threshold, 0.9, f32([0.3, 0.6]))


@case("a square batch (B = V), a single row, and vocabularies of one and two tokens")
def square_and_single(ctx):
    square = peaked(ctx, 6, 6)
    for probability in (0.4, 0.7):
        threshold = pick_threshold(ctx, square)
        check_fixed(ctx, square, threshold, probability, coins(ctx, 6, probability))
    single = peaked(ctx, 1, 10)
    check_fixed(ctx, single, pick_threshold(ctx, single), 1.0, f32([0.5]))
    check_fixed(ctx, f32([[3.5], [-200.0]]), 0.3, 1.0, f32([0.0, 0.5]))
    check_fixed(ctx, log_probs([[0.7, 0.3], [0.2, 0.8], [0.95, 0.05]]), 0.1, 1.0, f32([0.1] * 3))


@case("a vocabulary of 2000 tokens")
def large_vocabulary(ctx):
    logits = ctx.randn(4, 2000, scale=4.0)
    for probability in (0.5, 1.0):
        threshold = pick_threshold(ctx, logits, 0.01, 0.1)
        check_fixed(ctx, logits, threshold, probability, coins(ctx, 4, probability))


@case("how often XTC acts: 20,000 rows, probability 0.75, three tied top choices")
def acts_with_high_probability(ctx):
    acts_at_rate(ctx, f32([0.5, 2.0, 2.0, 2.5, -INF, 2.0]), 0.1, 0.75)
