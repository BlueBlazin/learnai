"""Tests for gumbel-max-sample.

Numeric cases pin down how the uniforms select a token, by comparing with the oracle.
The token is an argmax of float32 scores, and frameworks round `-log(-log(u))`
differently in its last bits, so `draw` and `check_tokens` keep the best score of every
row at least MARGIN above the second best (checked in float64), and at least
EDGE_MARGIN where a uniform of exactly 0 or 1 is involved: there the noise depends on
the clamp constants and on how accurately a framework evaluates log next to 1.

The statistical cases check what the function is for: with independent uniforms, the
tokens follow the softmax of the logits. There are three statistical checks, each with
the default delta of 1e-10, which keeps the false-failure rate of a correct solution
below 1e-9. Each is a single call with N rows.
"""

import numpy as np

from learnai.judge import case, check

FN = "gumbel_sample"
INF = np.inf
MARGIN = 1e-3
EDGE_MARGIN = 1.0
U_MIN = np.float32(1.1754944e-38)
U_MAX = np.float32(0.99999994)
N = 20_000


def f32(values):
    return np.array(values, dtype=np.float32)


def scores(logits, uniforms):
    u = np.clip(f32(uniforms), U_MIN, U_MAX).astype(np.float64)
    return np.asarray(logits, dtype=np.float64) - np.log(-np.log(u))


def gaps(logits, uniforms):
    """Per row: best score minus second best, divided by the margin that row needs."""
    s = scores(logits, uniforms)
    order = np.argsort(-s, axis=-1)
    if s.shape[1] == 1:
        return np.full(len(s), INF)
    best = np.take_along_axis(s, order[:, :1], axis=1)[:, 0]
    second = np.take_along_axis(s, order[:, 1:2], axis=1)[:, 0]
    u = f32(uniforms)
    edge = np.take_along_axis((u <= 0) | (u >= 1), order[:, :2], axis=1)
    edge[:, 1] &= np.isfinite(second)  # a masked runner-up loses whatever its uniform is
    need = np.where(edge.any(axis=1), EDGE_MARGIN, MARGIN)
    with np.errstate(invalid="ignore"):
        return np.where(np.isfinite(second), (best - second) / need, INF)


def draw(ctx, logits):
    """Independent uniforms in [0, 1), one per logit, redrawn until no row is a near tie."""
    u = ctx.rng.random(logits.shape, dtype=np.float32)
    while True:
        close = gaps(logits, u) < 1
        if not close.any():
            return u
        u[close] = ctx.rng.random((int(close.sum()), logits.shape[1]), dtype=np.float32)


def call(ctx, logits, uniforms):
    """Call the solution and check that it returns one unmasked token index per row."""
    tokens = np.asarray(ctx.call(FN, logits, uniforms))
    detail = f"logits = {logits}\nuniforms = {uniforms}\nresult = {tokens}"
    check.shape(tokens, (len(logits),), label=FN)
    check.that(
        tokens.dtype.kind in "iu",
        f"{FN}: expected an integer tensor of token indices, got dtype {tokens.dtype}",
    )
    check.that(
        bool(((tokens >= 0) & (tokens < logits.shape[1])).all()),
        f"{FN}: a returned index is outside [0, V)",
        detail,
    )
    chosen = np.take_along_axis(logits, tokens[:, None].astype(np.int64), axis=1)[:, 0]
    check.that(bool(np.isfinite(chosen).all()), f"{FN}: returned a masked token", detail)
    return tokens


def check_tokens(ctx, logits, uniforms):
    logits, uniforms = f32(logits), f32(uniforms)
    if gaps(logits, uniforms).min() < 1:  # an authoring error, not a solution failure
        raise RuntimeError("ambiguous test input: two scores of a row are within the margin")
    tokens = call(ctx, logits, uniforms)
    check.equal(tokens, ctx.expected(FN, logits, uniforms), label="tokens")
    return tokens


def softmax(row):
    x = np.asarray(row, dtype=np.float64)
    e = np.exp(x - x[np.isfinite(x)].max())
    return e / e.sum()


def follows_softmax(ctx, row):
    logits = np.tile(f32(row), (N, 1))
    uniforms = ctx.rng.random(logits.shape, dtype=np.float32)
    tokens = call(ctx, logits, uniforms)
    check.distribution(tokens, softmax(row), label="sampled token")


@case(
    "the worked example: noise beats a larger logit; a masked token with u = 1; u = 0",
    public=True,
)
def worked_example(ctx):
    logits = f32([[1.0, 2.0, 0.0, -INF], [0.0, 0.0, 0.0, 0.0], [-INF, 3.0, -INF, -INF]])
    uniforms = f32([[0.9, 0.3, 0.5, 0.99], [0.2, 0.6, 0.95, 0.5], [0.7, 0.0, 1.0, 0.5]])
    tokens = call(ctx, logits, uniforms)
    check.equal(tokens, [0, 2, 1], label="tokens")


@case("a random batch: every entry has its own uniform number", public=True)
def random_batch(ctx):
    # Catches noise shared between rows, one logarithm, and a flipped sign.
    logits = ctx.randn(6, 8, scale=1.5)
    for _ in range(4):
        check_tokens(ctx, logits, draw(ctx, logits))


@case(
    "the ends of the interval: u = 1 on masked tokens, u = 0 on the only unmasked token",
    public=True,
)
def interval_ends(ctx):
    # Catches a missing upper clamp (-inf + inf = NaN, which argmax returns) and a
    # missing lower clamp (every score -inf, argmax returns index 0, a masked token).
    logits = f32(
        [
            [-INF, 2.0, -INF, 0.5, -INF, 1.0],
            [-INF, -INF, -INF, 4.0, -INF, -INF],
            [-INF, -INF, 1.5, -INF, -INF, -3.0],
            [0.0, 1.0, -INF, 2.0, -INF, 3.0],
        ]
    )
    uniforms = f32(
        [
            [1.0, 0.5, 1.0, 0.4, 1.0, 0.6],
            [0.9, 1.0, 0.5, 0.0, 1.0, 0.3],
            [1.0, 0.2, 0.0, 0.7, 1.0, 0.0],
            [0.3, 0.4, 1.0, 0.5, 1.0, 0.6],
        ]
    )
    check_tokens(ctx, logits, uniforms)


@case("the samples follow the softmax of the logits: 20,000 rows, two tokens masked", public=True)
def follows_distribution(ctx):
    row = ctx.randn(8, scale=1.2)
    row[2] = row[7] = -INF
    follows_softmax(ctx, row)


@case("unnormalized logits far from zero: rows shifted by -900 and +700")
def shifted_logits(ctx):
    # At a magnitude of 1000 a float32 score resolves 6e-5, so the rows are drawn with
    # the margin of the unshifted rows and shifted by exactly representable amounts.
    logits = (ctx.randn(6, 8, scale=2.0)).astype(np.float32)
    uniforms = draw(ctx, logits)
    base = check_tokens(ctx, logits, uniforms)
    for shift in (-900.0, 700.0):
        shifted = (logits + np.float32(shift)).astype(np.float32)
        if gaps(shifted, uniforms).min() < 20:
            continue
        check.equal(check_tokens(ctx, shifted, uniforms), base, label="tokens after a shift")


@case("a finite token that draws u = 0 or u = 1 among other finite tokens")
def finite_extremes(ctx):
    # u = 0 is the smallest noise (-4.47), not -inf: the token still wins when its logit
    # leads by more than the others' noise. u = 1 is the largest noise (16.64).
    logits = f32(
        [
            [20.0, 0.0, -INF, 1.0, 0.5, -1.0],
            [0.0, 1.0, 2.0, -INF, -2.0, 0.5],
            [-3.0, -INF, 0.0, 1.0, 2.0, -1.0],
            [-INF, -INF, -INF, -INF, -INF, 7.0],
        ]
    )
    uniforms = f32(
        [
            [0.0, 0.5, 1.0, 0.6, 0.4, 0.3],
            [0.5, 0.4, 0.0, 1.0, 1.0, 0.6],
            [1.0, 1.0, 0.5, 0.0, 0.3, 0.4],
            [1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
        ]
    )
    check_tokens(ctx, logits, uniforms)


@case("a peaked row with rare and masked tokens: 20,000 samples follow its softmax")
def follows_peaked_distribution(ctx):
    with np.errstate(divide="ignore"):
        row = np.log(f32([0.0, 0.9, 0.0, 0.05, 0.03, 0.012, 0.005, 0.003])) + np.float32(40.0)
    follows_softmax(ctx, row)


@case("a nearly flat row: 20,000 samples follow its softmax")
def follows_flat_distribution(ctx):
    follows_softmax(ctx, ctx.randn(8, scale=0.3) - np.float32(5.0))


@case("a square batch (B = V), a single row, and a vocabulary of one token")
def square_and_single(ctx):
    square = ctx.randn(6, 6, scale=2.0)
    square[1, 0] = square[3, 3] = square[5, 5] = -INF
    for _ in range(3):
        check_tokens(ctx, square, draw(ctx, square))
    single = ctx.randn(1, 12, scale=2.0)
    check_tokens(ctx, single, draw(ctx, single))
    check_tokens(ctx, f32([[3.5], [-200.0], [0.0]]), f32([[0.0], [1.0], [0.5]]))


@case("a vocabulary of 1000 tokens with half of it masked")
def large_vocabulary(ctx):
    logits = ctx.randn(8, 1000, scale=3.0)
    logits[:, ctx.rng.random(1000) < 0.5] = -INF
    for _ in range(2):
        check_tokens(ctx, logits, draw(ctx, logits))
