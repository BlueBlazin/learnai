"""Tests for categorical-sample.

Numeric cases pin down how the uniform number selects a token, by comparing with the
oracle. A token changes where `u * total` crosses a cumulative sum, and float32
cumulative sums differ in their last bits between frameworks, so `uniforms` and
`check_tokens` keep every threshold at least MARGIN away from every cumulative sum
(checked in float64). The only exceptions are inputs whose arithmetic is exact in
float32: u = 0, probabilities that are multiples of 1/8, and one-hot rows.

The statistical cases check what the function is for: with independent uniforms, the
tokens follow the distribution. There are two statistical checks, each with the default
delta of 1e-10, which keeps the false-failure rate of a correct solution below 1e-9.
"""

import numpy as np

from whetstone.judge import case, check

FN = "sample"
MARGIN = 1e-4


def f32(values):
    return np.array(values, dtype=np.float32)


def gaps(probs, u):
    """Per row: the distance from u * total to the nearest cumulative sum, over the total.

    A row with u = 0 counts as infinitely far: its threshold is exactly 0, and "the
    first cumulative sum above 0" does not depend on rounding.
    """
    cdf = np.cumsum(np.asarray(probs, dtype=np.float64), axis=-1)
    total = cdf[:, -1:]
    u = np.asarray(u, dtype=np.float64)
    gap = (np.abs(cdf - u[:, None] * total) / total).min(axis=-1)
    return np.where(u == 0, np.inf, gap)


def uniforms(ctx, probs, low=0.0, high=1.0):
    """One uniform per row in [low, high), redrawn until it is clear of every boundary."""
    u = ctx.uniform(len(probs), low=low, high=high)
    while True:
        close = gaps(probs, u) < MARGIN
        if not close.any():
            return u
        u[close] = ctx.uniform(int(close.sum()), low=low, high=high)


def call(ctx, probs, u):
    """Call the solution and check that it returns one valid token index per row."""
    tokens = np.asarray(ctx.call(FN, probs, u))
    detail = f"probs = {probs}\nu = {u}\nresult = {tokens}"
    check.shape(tokens, (len(probs),), label=FN)
    check.that(
        tokens.dtype.kind in "iu",
        f"{FN}: expected an integer tensor of token indices, got dtype {tokens.dtype}",
    )
    check.that(
        bool(((tokens >= 0) & (tokens < probs.shape[1])).all()),
        f"{FN}: a returned index is outside [0, V)",
        detail,
    )
    chosen = np.take_along_axis(probs, tokens[:, None].astype(np.int64), axis=1)[:, 0]
    check.that(bool((chosen > 0).all()), f"{FN}: returned a token whose probability is 0", detail)
    return tokens


def check_tokens(ctx, probs, u, exact=False):
    probs, u = f32(probs), f32(u)
    if not exact and gaps(probs, u).min() < MARGIN:  # an authoring error
        raise RuntimeError("ambiguous test input: u * total is within MARGIN of a boundary")
    tokens = call(ctx, probs, u)
    check.equal(tokens, ctx.expected(FN, probs, u), label="tokens")
    return tokens


def distribution(ctx, vocab, concentration=1.0, zeros=()):
    """A random float32 probability vector with exact zeros at the given positions."""
    p = ctx.rng.dirichlet(np.full(vocab, concentration))
    p[list(zeros)] = 0.0
    return (p / p.sum()).astype(np.float32)


def exact(probs):
    probs = np.asarray(probs, dtype=np.float64)
    return probs / probs.sum()


@case(
    "the worked example: a plain row, u = 0 with a leading zero, a row that sums to 0.999",
    public=True,
)
def worked_example(ctx):
    probs = f32([[0.1, 0.4, 0.2, 0.3], [0.0, 0.5, 0.0, 0.5], [0.25, 0.25, 0.499, 0.0]])
    u = f32([0.55, 0.0, 0.9995])
    tokens = call(ctx, probs, u)
    check.equal(tokens, [2, 1, 2], label="tokens")


@case("a random batch: every row has its own distribution and its own u", public=True)
def random_batch(ctx):
    # Catches one uniform shared by the batch, and 1 - u used in place of u.
    probs = np.stack([distribution(ctx, 8) for _ in range(6)])
    for _ in range(3):
        check_tokens(ctx, probs, uniforms(ctx, probs))


@case(
    "exact boundaries: u = 0 skips leading zero-probability tokens; u on a boundary moves on",
    public=True,
)
def exact_boundaries(ctx):
    # Every probability is a multiple of 1/8 and every row sums to exactly 1, so the
    # cumulative sums and thresholds are exact in float32 in any framework. Catches
    # `>=` in place of `>`.
    probs = f32(
        [
            [0.0, 0.0, 0.5, 0.0, 0.5, 0.0],
            [0.25, 0.25, 0.0, 0.5, 0.0, 0.0],
            [0.125, 0.0, 0.375, 0.25, 0.0, 0.25],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    for u in (
        [0.0, 0.0, 0.0, 0.0],
        [0.5, 0.25, 0.125, 0.5],
        [0.25, 0.5, 0.5, 0.875],
        [0.75, 0.75, 0.75, 0.0],
    ):
        check_tokens(ctx, probs, u, exact=True)


@case("the samples follow the distribution: 4000 rows with independent uniforms", public=True)
def follows_distribution(ctx):
    p = distribution(ctx, 8, zeros=(2, 7))
    u = ctx.rng.random(4000, dtype=np.float32)
    tokens = call(ctx, np.tile(p, (4000, 1)), u)
    check.distribution(tokens, exact(p), label="sampled token")


@case("rows that sum to 0.999: a u above the row total still selects the last token")
def short_rows(ctx):
    # Catches comparing the cumulative sums with the raw u: none of them exceeds it,
    # and the count runs past the end of the row (index V).
    probs = f32([[0.3, 0.3, 0.399], [0.5, 0.2, 0.299], [0.111, 0.222, 0.666]])
    for u in ([0.9995, 0.9993, 0.9996], [0.9997, 0.45, 0.9992], [0.1, 0.9995, 0.2]):
        check_tokens(ctx, probs, u)


@case("rows that sum to 0.999 and end in zero-probability tokens")
def short_rows_trailing_zeros(ctx):
    # Catches the usual repairs of the previous failure: clamping the index to V - 1
    # (a zero-probability token) and an argmax over an all-False mask (token 0).
    probs = f32(
        [
            [0.3, 0.3, 0.399, 0.0, 0.0],
            [0.0, 0.5, 0.499, 0.0, 0.0],
            [0.2, 0.0, 0.3, 0.499, 0.0],
            [0.0, 0.0, 0.0, 0.999, 0.0],
        ]
    )
    for u in (
        [0.9995, 0.9993, 0.9996, 0.9994],
        [0.9997, 0.2, 0.9992, 0.0],
        [0.5, 0.9995, 0.1, 0.5],
    ):
        check_tokens(ctx, probs, u)


@case("rows that sum to 1.001: the uniform is scaled by the row total, not compared raw")
def long_rows(ctx):
    probs = f32([[0.3, 0.3, 0.401, 0.0], [0.6, 0.001, 0.2, 0.2], [0.0, 0.25, 0.25, 0.501]])
    for u in ([0.9995, 0.9993, 0.9996], [0.3, 0.6, 0.4], [0.7, 0.2, 0.05]):
        check_tokens(ctx, probs, u)


@case("one-hot rows and a vocabulary of one token: the answer does not depend on u")
def certain_rows(ctx):
    probs = f32(np.eye(5)[[3, 0, 4, 1]])
    for u in ([0.0, 0.0, 0.0, 0.0], [0.3, 0.9, 0.5, 0.7], [0.999, 0.999, 0.999, 0.999]):
        check_tokens(ctx, probs, u, exact=True)
    check_tokens(ctx, f32([[1.0], [1.0], [0.999]]), [0.0, 0.5, 0.9995], exact=True)


@case("a square batch (B = V), a single row, and peaked rows")
def square_and_single(ctx):
    square = np.stack([distribution(ctx, 6, concentration=0.3) for _ in range(6)])
    for _ in range(3):
        check_tokens(ctx, square, uniforms(ctx, square))
    single = distribution(ctx, 12, zeros=(0, 5, 11))[None]
    for _ in range(3):
        check_tokens(ctx, single, uniforms(ctx, single))


@case("a vocabulary of 1000 tokens, with u spread over [0, 1) and u close to 1")
def large_vocabulary(ctx):
    probs = np.stack([distribution(ctx, 1000, concentration=0.2) for _ in range(8)])
    check_tokens(ctx, probs, uniforms(ctx, probs))
    check_tokens(ctx, probs, uniforms(ctx, probs, low=0.97))


@case("a peaked distribution with rare and impossible tokens: 4000 samples follow it")
def follows_peaked_distribution(ctx):
    # Rows sum to 0.999 here: the target is the normalized row.
    p = f32([0.0, 0.9, 0.0, 0.05, 0.03, 0.01, 0.006, 0.003, 0.0, 0.0])
    u = ctx.rng.random(4000, dtype=np.float32)
    tokens = call(ctx, np.tile(p, (4000, 1)), u)
    check.distribution(tokens, exact(p), label="sampled token")
