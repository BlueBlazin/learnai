"""Tests for typical-acceptance.

Every case is exact. The expected pair is computed here, in float64 numpy, straight
from the definition, and `margins_ok` enforces what the statement promises: no draft
token's probability is within a relative 1e-3 of its threshold (so float32 rounding
cannot change a decision), and the row that supplies `next_token` has its two largest
entries at least 1e-3 apart.
The one exception is the equality case, whose numbers are dyadic and exact.

JAX compiles per shape, so the cases share (4, 4), (5, 8), and (9, 32).
"""

import numpy as np

from learnai.judge import case, check

FN = "typical_accept"

# The worked example of the statement.
P = np.array(
    [
        [0.70, 0.10, 0.10, 0.10],
        [0.25, 0.25, 0.25, 0.25],
        [0.40, 0.30, 0.24, 0.06],
        [0.05, 0.05, 0.85, 0.05],
    ],
    dtype=np.float32,
)


def thresholds(p, epsilon, delta):
    """min(epsilon, delta * exp(-H)) of every row, in float64, with 0 log 0 = 0."""
    p = p.astype(np.float64)
    entropy = -np.sum(p * np.log(np.where(p > 0, p, 1.0)), axis=-1)
    return np.minimum(epsilon, delta * np.exp(-entropy))


def definition(p, draft, epsilon, delta):
    """(n_accepted, next_token): stop at the first draft token that fails the test."""
    tau = thresholds(p, epsilon, delta)
    n = 0
    while n < len(draft) and float(p[n, draft[n]]) > tau[n]:
        n += 1
    return n, int(p[n].argmax())


def margins_ok(p, draft, epsilon, delta):
    tau = thresholds(p, epsilon, delta)
    prob = p[np.arange(len(draft)), draft].astype(np.float64)
    clear = np.abs(prob - tau[: len(draft)]) >= 1e-3 * tau[: len(draft)]
    # The row that supplies next_token must have a clear argmax.
    top = np.sort(p[definition(p, draft, epsilon, delta)[0]].astype(np.float64))[-2:]
    return bool(clear.all() and top[1] - top[0] >= 1e-3)


def verify(ctx, p, draft, epsilon, delta, expected=None, *, exact=False):
    p = np.asarray(p, dtype=np.float32)
    draft = np.asarray(draft, dtype=np.int64)
    if p.shape[0] != len(draft) + 1 or abs(float(p.sum(-1).max()) - 1) > 1e-4:
        raise RuntimeError("bad test input: wrong number of rows, or rows that do not sum to 1")
    if not exact and not margins_ok(p, draft, epsilon, delta):  # an authoring error
        raise RuntimeError("bad test input: a probability too close to its threshold")
    result = ctx.call(FN, p, draft, epsilon, delta)
    check.that(
        isinstance(result, (tuple, list)) and len(result) == 2,
        f"{FN}: expected an (n_accepted, next_token) pair, got {type(result).__name__}",
    )
    values = [np.asarray(value) for value in result]
    check.that(
        all(value.size == 1 and value.dtype.kind in "iu" for value in values),
        f"{FN}: n_accepted and next_token must each be a single integer",
        f"got {result!r}",
    )
    got = tuple(int(value.reshape(())) for value in values)
    want = definition(p, draft, epsilon, delta)
    if expected is not None and expected != want:  # an authoring error
        raise RuntimeError(f"the literal {expected} disagrees with the definition {want}")
    oracle = tuple(int(v) for v in ctx.expected(FN, p, draft, epsilon, delta))
    if oracle != want:  # an authoring error
        raise RuntimeError(f"the oracle {oracle} disagrees with the definition {want}")
    tau = thresholds(p, epsilon, delta)
    prob = p[np.arange(len(draft)), draft]
    detail = (
        f"call: {FN}(p, draft_tokens, epsilon={epsilon}, delta={delta})\n"
        f"  draft_tokens = {draft.tolist()}\n"
        f"  p of each draft token   = {np.round(prob, 5).tolist()}\n"
        f"  threshold of each row   = {np.round(tau[:-1], 5).tolist()}\n"
        f"  argmax of each row of p = {p.argmax(axis=-1).tolist()}\n"
        f"expected (n_accepted, next_token) = {want}\ngot {got}"
    )
    check.that(got[0] == want[0], f"{FN}: wrong number of accepted tokens", detail)
    check.that(got[1] == want[1], f"{FN}: wrong next token", detail)
    return got


def rows(ctx, n, vocab, concentration):
    """n probability rows; a small concentration gives peaked rows, a large one flat rows."""
    alpha = np.broadcast_to(np.asarray(concentration, dtype=np.float64), (n,))
    p = np.stack([ctx.rng.dirichlet(np.full(vocab, a)) for a in alpha])
    return (p / p.sum(-1, keepdims=True)).astype(np.float32)


def clear_case(ctx, k, vocab, epsilon, delta, concentration, draft_from="p"):
    """A random (p, draft) that satisfies the margins; redrawn until it does."""
    for _ in range(200):
        p = rows(ctx, k + 1, vocab, concentration)
        if draft_from == "p":  # tokens the target finds plausible
            draft = np.array([ctx.rng.choice(vocab, p=row / row.sum()) for row in p[:k]])
        elif draft_from == "argmax":
            draft = p[:k].argmax(-1)
        else:
            draft = ctx.rng.integers(0, vocab, size=k)
        if margins_ok(p, draft, epsilon, delta):
            return p, draft
    raise RuntimeError("could not draw a clear input")


@case("the worked example: three drafts over the same four rows", public=True)
def worked_example(ctx):
    verify(ctx, P, [1, 3, 3], 0.09, 0.3, expected=(2, 0))
    verify(ctx, P, [1, 3, 2], 0.09, 0.3, expected=(3, 2))
    verify(ctx, P, [1, 3, 3], 0.3, 0.3, expected=(0, 0))


# Rejects greedy_only (accepted tokens are not the argmax) and no_target_token.
@case("plausible tokens that are not the target's argmax are accepted", public=True)
def non_argmax_accepted(ctx):
    for _ in range(4):
        p = rows(ctx, 5, 8, 2.0)
        order = np.argsort(-p[:4], axis=-1)
        draft = order[:, 1]  # the second most likely token of every row
        if margins_ok(p, draft, 0.09, 0.3):
            verify(ctx, p, draft, 0.09, 0.3)
    p = rows(ctx, 5, 8, 2.0)
    verify(ctx, p, p[:4].argmax(-1), 0.09, 0.3, expected=(4, int(p[4].argmax())))


# Rejects continues_after_failure and threshold_from_next_row.
@case("a failure in the middle: later tokens that would pass are not counted", public=True)
def stops_at_first_failure(ctx):
    for position in (0, 1, 2, 3):
        for _ in range(200):
            p = rows(ctx, 5, 8, [0.3, 5.0, 0.3, 5.0, 1.0])
            draft = p[:4].argmax(-1)
            draft[position] = p[position].argmin()  # an implausible token at one position
            tau = thresholds(p, 0.09, 0.3)[:4]
            passes = p[np.arange(4), draft] > tau
            if margins_ok(p, draft, 0.09, 0.3) and passes.sum() == 3 and not passes[position]:
                break
        else:  # an authoring error
            raise RuntimeError("could not draw an input with exactly one failing token")
        verify(ctx, p, draft, 0.09, 0.3)


# Rejects entropy_in_bits and max_for_min: flat rows put delta * exp(-H) below epsilon.
@case("uncertain rows lower the bar: the entropy term is the smaller one", public=True)
def entropy_gate(ctx):
    flat = np.full((5, 8), 0.125, dtype=np.float32)
    flat[:, 0], flat[:, 1] = 0.14, 0.11  # H is close to ln 8, so the threshold is about 0.0376
    verify(ctx, flat, [1, 2, 3, 4], 0.09, 0.3, expected=(4, 0))
    # 0.06 is under the threshold in nats (0.0869) and over it in bits (0.0502).
    verify(ctx, P, [0, 0, 3], 0.09, 0.3, expected=(2, 0))
    p = rows(ctx, 5, 8, 20.0)
    draft = np.argsort(p[:4], axis=-1)[:, 2]
    if margins_ok(p, draft, 0.2, 0.5):
        verify(ctx, p, draft, 0.2, 0.5)


# Rejects accepts_on_equality.
@case("a probability exactly equal to the threshold is not accepted", public=True)
def strict_inequality(ctx):
    p = np.array(
        [[0.25, 0.75, 0.0, 0.0], [0.5, 0.25, 0.25, 0.0], [0.125, 0.0, 0.875, 0.0], [0, 0, 0, 1]],
        dtype=np.float32,
    )
    # Every threshold is epsilon (delta * exp(-H) is larger), and it is exact in float32.
    verify(ctx, p, [0, 0, 0], 0.25, 1.0, expected=(0, 1), exact=True)
    verify(ctx, p, [1, 1, 0], 0.25, 1.0, expected=(1, 0), exact=True)
    verify(ctx, p, [1, 0, 0], 0.125, 1.0, expected=(2, 2), exact=True)


# Rejects zero_probability_nan.
@case("rows with zero probabilities: masked tokens, and a one-hot row")
def zero_probabilities(ctx):
    p = rows(ctx, 5, 8, 1.0)
    p[:, 5:] = 0.0
    p[1] = 0.0
    p[1, 2] = 1.0  # a one-hot row: entropy 0, threshold min(epsilon, delta)
    p = p / p.sum(-1, keepdims=True)
    draft = p[:4].argmax(-1)
    verify(ctx, p, draft, 0.09, 0.3, expected=(4, int(p[4].argmax())))
    draft[2] = 6  # a token of probability 0 is never accepted
    verify(ctx, p, draft, 0.09, 0.3, expected=(2, int(p[2].argmax())))


@case("epsilon and delta at their extremes")
def threshold_extremes(ctx):
    for epsilon, delta in ((1.0, 1.0), (1.0, 0.05), (0.01, 1.0), (0.5, 0.5), (0.02, 0.02)):
        p, draft = clear_case(ctx, 4, 8, epsilon, delta, 1.0)
        verify(ctx, p, draft, epsilon, delta)


# Rejects one_entropy_for_all_rows.
@case("peaked and flat rows alternate: every position has its own threshold")
def per_row_thresholds(ctx):
    for _ in range(4):
        p, draft = clear_case(ctx, 8, 32, 0.09, 0.3, [0.05, 50.0] * 4 + [1.0], draft_from="p")
        verify(ctx, p, draft, 0.09, 0.3)


@case("a single draft token, and the largest sizes")
def sizes(ctx):
    p = rows(ctx, 2, 8, 1.0)
    for token in (int(p[0].argmax()), int(p[0].argmin())):
        if margins_ok(p, np.array([token]), 0.09, 0.3):
            verify(ctx, p, [token], 0.09, 0.3)
    p, draft = clear_case(ctx, 16, 64, 0.09, 0.3, 0.5)
    verify(ctx, p, draft, 0.09, 0.3)


@case("random distributions, drafts, and thresholds")
def random_sweep(ctx):
    sources = ("p", "argmax", "uniform")
    seen = set()
    for trial in range(24):
        epsilon = float(ctx.rng.choice([0.05, 0.09, 0.2, 0.4]))
        delta = float(ctx.rng.choice([0.1, 0.3, 0.6, 1.0]))
        k, vocab = (4, 8) if trial % 2 else (8, 32)
        concentration = 10.0 ** ctx.rng.uniform(-1.5, 1.5, size=k + 1)
        p, draft = clear_case(ctx, k, vocab, epsilon, delta, concentration, sources[trial % 3])
        seen.add(verify(ctx, p, draft, epsilon, delta)[0])
    if len(seen) < 4:  # an authoring error
        raise RuntimeError("the sweep does not cover enough different outcomes")
