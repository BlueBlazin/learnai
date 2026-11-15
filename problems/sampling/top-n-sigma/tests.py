"""Tests for top-n-sigma. Every case compares the solution with the oracle.

Whether a token is kept depends on comparing its logit with `max - n * std`. Float32
frameworks round the standard deviation differently in its last bits, so every case
keeps all finite logits at least MARGIN away from the cutoff (checked in float64 by
`margin`). The exception is exact: when the standard deviation of a row is 0, the
entries equal to the maximum sit on the cutoff, in every framework.
"""

import numpy as np

from whetstone.judge import case, check

FN = "top_n_sigma_filter"
INF = np.inf
MARGIN = 1e-3


def f32(rows):
    return np.array(rows, dtype=np.float32)


def stats(logits):
    """Per row, in float64: the maximum and the population std of the finite logits."""
    x = np.asarray(logits, dtype=np.float64)
    finite = np.isfinite(x)
    count = finite.sum(axis=-1, keepdims=True)
    mean = np.where(finite, x, 0.0).sum(axis=-1, keepdims=True) / count
    var = np.where(finite, (np.where(finite, x, 0.0) - mean) ** 2, 0.0).sum(-1, keepdims=True)
    return np.where(finite, x, -INF).max(axis=-1, keepdims=True), np.sqrt(var / count)


def margin(logits, n):
    """Smallest distance between a finite logit and the cutoff `max - n * std`."""
    if n <= 0:
        return INF
    x = np.asarray(logits, dtype=np.float64)
    top, std = stats(x)
    gap = np.abs(x - (top - n * std))
    gap[~np.isfinite(x)] = INF
    gap[(std == 0) & (x == top)] = INF  # exactly on the cutoff, and exactly representable
    return float(gap.min())


def pick_n(ctx, logits, low=0.3, high=3.0, clearance=MARGIN):
    """A random `n` in [low, high) that leaves no logit of any row near its cutoff."""
    while True:
        n = float(f"{ctx.rng.uniform(low, high):.3g}")
        if margin(logits, n) >= clearance:
            return n


def check_fixed(ctx, logits, n):
    logits = f32(logits)
    if margin(logits, n) < MARGIN:  # an authoring error, not a solution failure
        raise RuntimeError("ambiguous test input: a logit is within MARGIN of the cutoff")
    return ctx.check(FN, logits, n)


def scaled(ctx, shape, scales):
    """Random rows, row i with standard deviation about scales[i]."""
    return ctx.randn(*shape) * f32(scales)[:, None]


@case(
    "the worked example: a row with a masked entry and a row with one outlier, n = 1", public=True
)
def worked_example(ctx):
    logits = f32([[2.0, 1.0, 0.0, -1.0, -2.0, -INF], [10.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    got = ctx.call(FN, logits, 1.0)
    keep = np.array([[1, 1, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0]], dtype=bool)
    check.allclose(got, np.where(keep, logits, -INF).astype(np.float32), label=FN, dtype=True)


@case("a random batch: rows with a small and a large spread, three values of n", public=True)
def random_batch(ctx):
    # Catches statistics shared by the batch, and a band below the mean.
    logits = scaled(ctx, (4, 10), [0.3, 1.5, 3.0, 12.0])
    logits += f32([[5.0], [-3.0], [0.0], [20.0]])
    for low, high in ((0.3, 0.8), (0.8, 1.6), (1.6, 3.0)):
        check_fixed(ctx, logits, pick_n(ctx, logits, low, high))


@case("masked logits (-inf) stay masked and count toward neither the mean nor the std", public=True)
def masked_logits(ctx):
    # Catches a std over the whole row (NaN) and sums divided by V instead of |F|.
    logits = scaled(ctx, (4, 10), [1.0, 2.0, 2.0, 4.0]) + f32([[6.0], [-8.0], [3.0], [0.0]])
    logits[0, 2] = logits[0, 5] = logits[2, 0] = -INF
    logits[1, 3:] = -INF
    logits[3, :6] = -INF
    for low, high in ((0.3, 1.0), (1.0, 2.0), (2.0, 3.0)):
        check_fixed(ctx, logits, pick_n(ctx, logits, low, high))


@case(
    "temperature does not change the kept set: the same rows divided by 0.5, 2, and 4",
    public=True,
)
def temperature_invariant(ctx):
    # Dividing by a power of two is exact and scales the margins with the logits, so n
    # is chosen with four times the clearance. Catches a band measured on probabilities.
    logits = scaled(ctx, (4, 10), [1.0, 2.0, 3.0, 5.0])
    n = pick_n(ctx, logits, 0.8, 2.0, clearance=4 * MARGIN)
    base = check_fixed(ctx, logits, n)
    for temperature in (0.5, 2.0, 4.0):
        tempered = (logits / np.float32(temperature)).astype(np.float32)
        got = check_fixed(ctx, tempered, n)
        check.that(
            np.array_equal(np.isfinite(got), np.isfinite(base)),
            f"{FN}: dividing every logit by a temperature changed which tokens are kept",
            f"temperature = {temperature}, n = {n}\nlogits = {tempered}\nresult = {got}",
        )


@case("the population standard deviation on short rows, where |F| and |F| - 1 differ most")
def population_std(ctx):
    # Row 0: std is sqrt(3) = 1.732 with divisor 4 and 2 with divisor 3; for n = 2.2 the
    # cutoffs are 0.19 and -0.4, on either side of the zeros. Catches the sample std.
    logits = f32([[4.0, 0.0, 0.0, 0.0], [0.0, 3.0, -3.0, 1.0], [-INF, 5.0, 2.0, -INF]])
    for n in (2.2, 1.25, 0.6):
        check_fixed(ctx, logits, n)


@case("rows with zero spread: equal logits, a single finite logit, a vocabulary of one")
def zero_spread(ctx):
    # The cutoff equals the maximum; `>=` keeps it. Catches `>` and a divisor of |F| - 1.
    logits = f32(
        [
            [2.5, 2.5, 2.5, 2.5],
            [-INF, -7.0, -INF, -INF],
            [0.0, 0.0, -INF, 0.0],
            [1.0, 3.0, 3.0, 1.0],
        ]
    )
    for n in (0.5, 1.5, 3.0):
        check_fixed(ctx, logits, n)
    for n in (0.4, 2.0):
        check_fixed(ctx, f32([[3.5], [-90.0]]), n)


@case("n <= 0 disables the filter: the logits come back unchanged")
def disabled(ctx):
    logits = scaled(ctx, (4, 10), [1.0, 2.0, 3.0, 5.0])
    logits[1, 4] = logits[3, 0] = -INF
    for n in (0.0, -1.0, -0.25):
        check_fixed(ctx, logits, n)


@case("rows shifted far from zero: only differences of logits matter")
def shifted_rows(ctx):
    logits = scaled(ctx, (4, 10), [0.5, 1.0, 2.0, 4.0])
    n = pick_n(ctx, logits, 0.6, 2.0)
    base = check_fixed(ctx, logits, n)
    for shift in (-64.0, 32.0, 80.0):
        shifted = (logits + np.float32(shift)).astype(np.float32)
        if margin(shifted, n) < MARGIN:
            continue
        got = check_fixed(ctx, shifted, n)
        check.that(
            np.array_equal(np.isfinite(got), np.isfinite(base)),
            f"{FN}: adding a constant to every logit of a row changed which tokens are kept",
            f"shift = {shift}, n = {n}\nlogits = {shifted}\nresult = {got}",
        )


@case("a square batch (B = V) and a single row, with masked entries")
def square_and_single(ctx):
    square = scaled(ctx, (6, 6), [0.5, 1.0, 2.0, 3.0, 5.0, 8.0])
    square[2, 1] = square[4, 4] = square[5, 0] = -INF
    for _ in range(3):
        check_fixed(ctx, square, pick_n(ctx, square))
    single = ctx.randn(1, 12, scale=2.0)
    single[0, 7] = -INF
    check_fixed(ctx, single, pick_n(ctx, single))


@case("a vocabulary of 2000 tokens with a third of it masked")
def large_vocabulary(ctx):
    logits = ctx.randn(2, 2000, scale=2.5)
    logits[0, ctx.rng.random(2000) < 0.33] = -INF
    check_fixed(ctx, logits, pick_n(ctx, logits, 1.0, 2.0))
    check_fixed(ctx, logits, pick_n(ctx, logits, 2.0, 3.0))
