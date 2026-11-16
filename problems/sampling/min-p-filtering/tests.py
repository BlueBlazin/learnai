"""Tests for min-p-filtering.

Numeric cases compare the solution with the oracle; the property case states the
definition in numpy.

Whether a token is kept depends on comparing its probability with `min_p` times the top
probability, which is the same as comparing its logit with `top logit + log(min_p)`. A
correct solution may do either, and float32 rounds the two differently, so every case
keeps all logits at least MARGIN away from that cutoff (checked in float64 by `margin`).
The exceptions are exact: `min_p = 0` keeps everything, and with `min_p = 1` a logit
equal to the top logit has exactly the top probability.
"""

import numpy as np

from learnai.judge import case, check

FN = "min_p_filter"
INF = np.inf
MARGIN = 1e-3


def margin(logits, min_p):
    """Smallest distance between a finite logit and the cutoff `top + log(min_p)`."""
    if min_p == 0:
        return INF
    x = np.asarray(logits, dtype=np.float64)
    gap = np.abs(x - (x.max(axis=-1, keepdims=True) + np.log(min_p)))
    gap[~np.isfinite(x)] = INF
    if min_p == 1:
        gap[x == x.max(axis=-1, keepdims=True)] = INF  # exact ties with the top token
    return float(gap.min())


def pick_min_p(ctx, logits, low=1e-3, high=0.9):
    """A random `min_p` in [low, high), log-uniform, unambiguous for every row."""
    while True:
        min_p = float(f"{np.exp(ctx.rng.uniform(np.log(low), np.log(high))):.3g}")
        if margin(logits, min_p) >= MARGIN:
            return min_p


def check_fixed(ctx, logits, min_p):
    logits = np.asarray(logits, dtype=np.float32)
    if margin(logits, min_p) < MARGIN:  # an authoring error, not a solution failure
        raise RuntimeError("ambiguous test input: a logit is within MARGIN of the cutoff")
    return ctx.check(FN, logits, min_p)


@case("the worked example: a confident row and a flat row, min_p = 0.2", public=True)
def worked_example(ctx):
    with np.errstate(divide="ignore"):
        logits = np.log(
            np.array(
                [[0.6, 0.3, 0.05, 0.03, 0.02, 0.0], [0.18, 0.18, 0.17, 0.16, 0.16, 0.15]],
                dtype=np.float32,
            )
        )
    got = ctx.call(FN, logits, 0.2)
    keep = np.array([[1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1]], dtype=bool)
    expected = np.where(keep, logits, -INF).astype(np.float32)
    check.allclose(got, expected, label=FN, dtype=True)


@case("a random batch: rows from flat to peaked keep different numbers of tokens", public=True)
def random_batch(ctx):
    # Catches a top probability taken over the whole batch, and an absolute cutoff.
    logits = ctx.randn(4, 10) * np.array([[0.3], [1.5], [3.0], [6.0]], dtype=np.float32)
    for low, high in ((0.02, 0.1), (0.1, 0.4), (0.4, 0.9)):
        check_fixed(ctx, logits, pick_min_p(ctx, logits, low, high))


@case("min_p = 0 keeps everything; min_p = 1 keeps only the top token and its ties", public=True)
def extreme_min_p(ctx):
    # Catches `>` in place of `>=`: with min_p = 1 the top token is exactly at the cutoff.
    logits = np.array(
        [
            [1.0, 3.0, 2.0, 3.0, -1.0, 0.5],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [-5.0, -2.5, -9.0, -2.5, -2.5, -7.0],
            [4.0, 1.0, -INF, 2.0, 3.0, -INF],
        ],
        dtype=np.float32,
    )
    check_fixed(ctx, logits, 0.0)
    check_fixed(ctx, logits, 1.0)
    check_fixed(ctx, logits, 0.5)


@case(
    "the cutoff depends on logit differences only: the same rows shifted by -800 and +500",
    public=True,
)
def shifted_rows(ctx):
    # Catches a ratio applied to logits: `min_p * top logit` changes with a shift, and
    # for a negative top logit it lies above the top logit.
    logits = ctx.randn(4, 8, scale=2.0)
    min_p = pick_min_p(ctx, logits, 0.05, 0.5)
    base = check_fixed(ctx, logits, min_p)
    for shift in (-800.0, 500.0, -3.0):
        shifted = (logits + np.float32(shift)).astype(np.float32)
        got = check_fixed(ctx, shifted, min_p)
        check.that(
            np.array_equal(np.isfinite(got), np.isfinite(base)),
            f"{FN}: adding a constant to every logit of a row changed which tokens are kept",
            f"shift = {shift}, min_p = {min_p}\nlogits = {shifted}\nresult = {got}",
        )


@case("masked logits (-inf) stay masked and do not affect the cutoff")
def masked_logits(ctx):
    logits = ctx.randn(4, 10, scale=2.0)
    logits[0, 2] = logits[0, 5] = logits[2, 0] = -INF
    logits[1, 1:] = -INF  # one unmasked token: it always survives
    logits[3, :8] = -INF
    for _ in range(3):
        check_fixed(ctx, logits, pick_min_p(ctx, logits))
    check_fixed(ctx, logits, 0.0)
    check_fixed(ctx, logits, 1.0)


@case("removed tokens are -inf, also where the logits are positive or exactly 0")
def removed_is_masked(ctx):
    # Catches filling removed entries with 0: that is what a removed probability looks
    # like, but as a logit it is an ordinary value, here larger than most of the row.
    logits = np.array(
        [[9.0, 0.0, 0.0, 8.5, 0.0, 2.0], [0.0, -4.0, 0.0, -6.0, 0.0, -0.5]], dtype=np.float32
    )
    for min_p in (0.3, 0.02, 0.9):
        check_fixed(ctx, logits, min_p)


@case("very small and very large min_p on rows with a wide spread of logits")
def wide_spread(ctx):
    logits = ctx.randn(4, 12) * np.array([[40.0], [12.0], [4.0], [150.0]], dtype=np.float32)
    for low, high in ((1e-4, 1e-3), (1e-3, 1e-2), (0.9, 0.999)):
        check_fixed(ctx, logits, pick_min_p(ctx, logits, low, high))


@case("a square batch (B = V), a single row, and a vocabulary of one token")
def square_and_single(ctx):
    square = ctx.randn(6, 6) * np.array([[0.5], [1.0], [2.0], [3.0], [5.0], [8.0]], np.float32)
    for _ in range(3):
        check_fixed(ctx, square, pick_min_p(ctx, square))
    single = ctx.randn(1, 12, scale=2.0)
    check_fixed(ctx, single, pick_min_p(ctx, single))
    for min_p in (0.0, 0.3, 1.0):
        check_fixed(ctx, np.array([[3.5], [-200.0]], dtype=np.float32), min_p)


@case("a vocabulary of 2000 tokens")
def large_vocabulary(ctx):
    logits = ctx.randn(2, 2000, scale=2.5)
    check_fixed(ctx, logits, pick_min_p(ctx, logits, 0.01, 0.05))
    check_fixed(ctx, logits, pick_min_p(ctx, logits, 0.2, 0.6))


@case("the definition: kept exactly when probability >= min_p * top probability; top survives")
def min_p_properties(ctx):
    shapes = [(4, 10), (6, 6), (1, 12)]  # shapes other cases already use, for JAX

    def prop(rng):
        shape = shapes[int(rng.integers(len(shapes)))]
        logits = ctx.randn(*shape, scale=float(10.0 ** rng.uniform(-0.5, 1.0)))
        logits = (logits + np.float32(rng.uniform(-50, 50))).astype(np.float32)
        logits[rng.random(shape) < 0.1] = -INF
        # One entry per row is set after masking, so every row keeps a finite logit.
        logits[np.arange(shape[0]), rng.integers(shape[1], size=shape[0])] = 0.5
        min_p = pick_min_p(ctx, logits, 1e-3, 0.95)
        out = ctx.call(FN, logits, min_p)
        check.shape(out, shape, label=FN)
        check.that(not np.isnan(out).any(), f"{FN}: the result contains NaN")
        x = logits.astype(np.float64)
        for row, (xi, o) in enumerate(zip(x, out, strict=True)):
            where = f"row {row}, min_p = {min_p}\nlogits = {xi}\nresult = {o}"
            kept = np.isfinite(o)
            check.that(
                bool(kept[int(np.argmax(xi))]),
                f"{FN}: the most likely token of a row was removed",
                where,
            )
            check.that(
                bool(((o == xi.astype(np.float32)) | np.isneginf(o)).all()),
                f"{FN}: an entry is neither its input logit nor -inf",
                where,
            )
            # p_i / p_top = exp(x_i - x_top): the ratio test, in log space and float64.
            should_keep = xi - xi.max() >= np.log(min_p)
            check.that(
                np.array_equal(kept, should_keep),
                f"{FN}: the kept tokens are not those with probability >= min_p * top probability",
                where,
            )

    check.forall(prop, trials=8)
