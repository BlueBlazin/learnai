"""Tests for top-p-filtering.

Numeric cases compare the solution with the oracle; the property case states what
nucleus filtering means without reference to any implementation.

Whether a token is kept depends on comparing a partial sum of probabilities with `p`.
Every case therefore keeps all partial sums at least `MARGIN` away from `p` (checked in
float64 by `margin`), so float32 rounding can never decide the outcome.
"""

import numpy as np

from whetstone.judge import case, check

MARGIN = 1e-3


def softmax64(logits):
    x = np.asarray(logits, dtype=np.float64)
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def margin(logits, p):
    """Smallest distance between `p` and a partial sum of the k < V largest probabilities."""
    ranked = -np.sort(-softmax64(logits), axis=-1)
    partial = np.cumsum(ranked, axis=-1)[:, :-1]
    return float(np.abs(partial - p).min()) if partial.size else 1.0


def pick_p(ctx, logits, low=0.05, high=0.98, min_margin=MARGIN):
    """A random `p` in [low, high) that is unambiguous for every row of `logits`."""
    while True:
        p = round(float(ctx.rng.uniform(low, high)), 4)
        if margin(logits, p) >= min_margin:
            return p


def check_fixed(ctx, logits, p, min_margin=MARGIN):
    logits = np.asarray(logits, dtype=np.float32)
    if margin(logits, p) < min_margin:  # an authoring error, not a solution failure
        raise RuntimeError(f"ambiguous test input: a partial sum is within {min_margin} of p")
    return ctx.check("top_p_filter", logits, p)


@case(
    "the worked example: probabilities [0.1, 0.4, 0.2, 0.3] and [0.7, 0.1, 0.1, 0.1]", public=True
)
def worked_example(ctx):
    logits = np.log(np.array([[1.0, 4.0, 2.0, 3.0], [7.0, 1.0, 1.0, 1.0]], dtype=np.float32))
    got = ctx.call("top_p_filter", logits, 0.6)
    expected = np.array([[0.0, 4 / 7, 0.0, 3 / 7], [1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    check.allclose(got, expected, label="top_p_filter", dtype=True)


@case("a random batch: each row keeps a different number of tokens", public=True)
def random_batch(ctx):
    # Catches a cutoff shared by the batch, and results left in sorted order.
    logits = ctx.randn(4, 10) * np.array([[0.5], [1.5], [3.0], [6.0]], dtype=np.float32)
    check_fixed(ctx, logits, pick_p(ctx, logits, 0.5, 0.9))


@case("the top token alone has more than p: it is the only survivor", public=True)
def dominant_token(ctx):
    logits = np.array([[0.0, 5.0, 1.0, -2.0], [3.0, 2.9, -1.0, 8.0]], dtype=np.float32)
    check_fixed(ctx, logits, 0.5)
    check_fixed(ctx, logits, 0.01)


@case("ties: equal probabilities are ranked by token index, lowest first", public=True)
def ties(ctx):
    # Probabilities [0.25, 0.25, 0.25, 0.25] and [0.125, 0.375, 0.125, 0.375]: p = 0.6
    # keeps tokens 0, 1, 2 of the first row; p = 0.3 keeps only token 1 of the second.
    flat = np.log(np.array([[1.0, 1.0, 1.0, 1.0], [1.0, 3.0, 1.0, 3.0]], dtype=np.float32))
    for p in (0.6, 0.3, 0.8):
        check_fixed(ctx, flat, p)
    # [0.125, 0.25, 0.125, 0.5] and [0.375, 0.125, 0.375, 0.125]: at p = 0.8 the third
    # token kept is one of a tied pair (token 0, not 2, in row 0; token 1, not 3, in row 1).
    logits = np.log(np.array([[1.0, 2.0, 1.0, 4.0], [3.0, 1.0, 3.0, 1.0]], dtype=np.float32))
    for p in (0.2, 0.6, 0.8, 0.95):
        check_fixed(ctx, logits, p)


@case("p = 1.0 keeps every token")
def keep_everything(ctx):
    logits = ctx.randn(4, 10)
    check_fixed(ctx, logits, 1.0)


@case("masked logits (-inf) are never kept")
def masked_logits(ctx):
    logits = ctx.randn(4, 10, scale=2.0)
    logits[0, 2] = logits[0, 5] = logits[2, 0] = -np.inf
    logits[1, 1:] = -np.inf  # one unmasked token: it gets probability 1
    for _ in range(3):
        check_fixed(ctx, logits, pick_p(ctx, logits))


@case("peaked and flat rows with logits of magnitude 1e3")
def large_logits(ctx):
    logits = ctx.randn(4, 10) * np.array([[1000.0], [100.0], [10.0], [0.01]], dtype=np.float32)
    for _ in range(3):
        check_fixed(ctx, logits, pick_p(ctx, logits))


@case("a square batch (B = V) and a batch of one row")
def square_and_single(ctx):
    # A square batch hides a filter applied along the wrong axis; one row hides
    # nothing, but exercises B = 1.
    square = ctx.randn(6, 6, scale=2.0)
    for _ in range(3):
        check_fixed(ctx, square, pick_p(ctx, square))
    single = ctx.randn(1, 12, scale=2.0)
    check_fixed(ctx, single, pick_p(ctx, single))


@case("a vocabulary of one token")
def single_token(ctx):
    logits = np.array([[3.5], [-200.0]], dtype=np.float32)
    check_fixed(ctx, logits, 0.3)
    check_fixed(ctx, logits, 1.0)


@case("a vocabulary of 2000 tokens with a small p and a large p")
def large_vocabulary(ctx):
    logits = ctx.randn(2, 2000, scale=2.5)
    check_fixed(ctx, logits, pick_p(ctx, logits, 0.1, 0.3, min_margin=1e-4), min_margin=1e-4)
    check_fixed(ctx, logits, pick_p(ctx, logits, 0.9, 0.99, min_margin=1e-4), min_margin=1e-4)


@case("the definition: the smallest set of most probable tokens with mass >= p, renormalized")
def nucleus_properties(ctx):
    shapes = [(2, 4), (4, 10), (6, 6), (1, 12)]  # few distinct shapes, for JAX

    def prop(rng):
        shape = shapes[int(rng.integers(len(shapes)))]
        logits = ctx.randn(*shape, scale=float(10.0 ** rng.uniform(-0.5, 0.8)))
        p = pick_p(ctx, logits)
        out = ctx.call("top_p_filter", logits, p)
        check.shape(out, shape, label="top_p_filter")
        check.finite(out, label="top_p_filter")
        check.that((out >= 0).all(), "top_p_filter: the result has negative entries")
        check.allclose(out.sum(axis=-1), np.ones(shape[0]), atol=1e-5, label="row sums")

        probs = softmax64(logits)
        for row, (pi, o) in enumerate(zip(probs, out, strict=True)):
            kept = o > 0
            where = f"row {row}, p = {p}\nprobabilities = {pi}\nresult = {o}"
            check.that(kept.any(), "top_p_filter: a row has no surviving token", where)
            mass = pi[kept].sum()
            check.that(
                mass >= p, "top_p_filter: the kept tokens have less than p of the mass", where
            )
            check.that(
                mass - pi[kept].min() < p,
                "top_p_filter: the kept set is not the smallest one (it reaches p without its "
                "least likely token)",
                where,
            )
            if not kept.all():
                check.that(
                    pi[kept].min() >= pi[~kept].max(),
                    "top_p_filter: a removed token is more probable than a kept one",
                    where,
                )
            check.allclose(o[kept], pi[kept] / mass, label="renormalized kept probabilities")

    check.forall(prop, trials=8)
