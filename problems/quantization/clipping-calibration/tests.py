"""Tests for clipping-calibration.

The answer is one of the candidates, so a case is only fair if float32 rounding cannot
change which candidate wins. Two kinds of input guarantee that:

- `separated` drops every candidate whose MSE is within 2 percent of the best one
  (other than the best itself), so the winner is clear by a margin far above float32
  error;
- the tie cases use 2 bits, small integer samples whose largest magnitude is a power of
  two, and candidates that are multiples of 1/8. Every limit, scale, quantized value,
  and squared error is then a small integer, exact in float32 whatever the order of
  operations, so tied candidates are tied bit for bit.
"""

import numpy as np

from learnai.judge import case, check

GAP = 1.02


def mse_table(samples, bits, candidates):
    """MSE of every candidate in float64, straight from the statement."""
    x = np.asarray(samples, dtype=np.float64).reshape(-1)
    qmax = 2 ** (bits - 1) - 1
    scale = (np.asarray(candidates, dtype=np.float64) * np.abs(x).max() / qmax)[:, None]
    x_hat = np.clip(np.round(x / scale), -qmax, qmax) * scale
    return ((x_hat - x) ** 2).mean(axis=1)


def separated(samples, bits, candidates):
    """The candidates without the runners-up that are within 2 percent of the best MSE."""
    candidates = np.asarray(candidates, dtype=np.float32)
    mse = mse_table(samples, bits, candidates)
    keep = mse >= mse.min() * GAP
    keep[np.argmin(mse)] = True
    return candidates[keep]


def check_clip(ctx, samples, bits, candidates, *, exact_ties=False):
    """Compare with the oracle; the result must be a 0-d float32 tensor."""
    samples = np.asarray(samples, dtype=np.float32)
    if not exact_ties:
        candidates = separated(samples, bits, candidates)
    candidates = np.asarray(candidates, dtype=np.float32)
    got = ctx.check("best_clip", samples, bits, candidates)
    check.shape(got, (), label="best_clip")
    return float(got), candidates


def heavy_tailed(ctx, *shape, df=3.0):
    """Student-t samples: a bell-shaped bulk with a few far outliers, like real weights."""
    return ctx.rng.standard_t(df, size=shape).astype(np.float32)


GRID = np.linspace(1.0, 0.3, 8, dtype=np.float32)  # 1.0, 0.9, ..., 0.3


@case("the worked example: six samples, 2 bits, candidates 1.0, 0.5, 0.25", public=True)
def worked_example(ctx):
    samples = np.array([0.8, -0.9, 1.1, -1.2, 0.6, 2.0], dtype=np.float32)
    candidates = np.array([1.0, 0.5, 0.25], dtype=np.float32)
    got = ctx.call("best_clip", samples, 2, candidates)
    check.allclose(got, np.float32(0.5), label="best_clip")
    check_clip(ctx, samples, 2, candidates)


# Rejects clipping_error_only (it always returns the largest range) and max_per_sample.
@case("heavy-tailed samples at 4 bits: clipping the outliers wins", public=True)
def outliers_reward_clipping(ctx):
    samples = heavy_tailed(ctx, 16, 64)
    got, _ = check_clip(ctx, samples, 4, GRID)
    check.that(got < 1.0, "with far outliers and 16 levels, some clipping must beat none")


# Rejects no_clamp (it always prefers the smallest range).
@case("uniform samples at 8 bits: nothing to gain from clipping", public=True)
def uniform_keeps_full_range(ctx):
    samples = ctx.uniform(16, 64, low=-2.0, high=2.0)
    got, _ = check_clip(ctx, samples, 8, GRID)
    check.that(got == 1.0, "clipping uniformly spread samples only adds error")


# Rejects ties_to_smaller and returns_first_minimum (the tie is listed smaller-first).
@case("an exact tie between 0.75 and 1.0 goes to the larger range", public=True)
def tie_with_full_range(ctx):
    # Squared errors are (5, 5, 29, 77) / 4 for fractions 1, 0.75, 0.5, 0.25.
    samples = np.array([8.0, -6.0, 7.0, 0.0], dtype=np.float32)
    ascending = np.array([0.25, 0.5, 0.75, 1.0], dtype=np.float32)
    for candidates in (ascending, ascending[::-1].copy()):
        got, _ = check_clip(ctx, samples, 2, candidates, exact_ties=True)
        check.that(got == 1.0, "a tie must go to the larger candidate")


# Rejects signed_max.
@case("the largest magnitude is negative")
def negative_peak(ctx):
    samples = heavy_tailed(ctx, 16, 64)
    samples = -np.abs(samples) * 3.0 + ctx.randn(16, 64, scale=0.5)
    check_clip(ctx, samples, 4, GRID)
    check_clip(ctx, np.array([0.8, -0.9, 1.1, -1.2, 0.6, -2.0], np.float32), 2, [1.0, 0.5, 0.25])


# Rejects max_per_sample: one loud row sets the range for everyone.
@case("samples of very different magnitudes share one range")
def loud_row(ctx):
    samples = ctx.randn(16, 64)
    samples[3] *= 12.0
    check_clip(ctx, samples, 4, GRID)
    samples = ctx.randn(16, 64) * ctx.uniform(16, 1, low=0.2, high=6.0)
    check_clip(ctx, samples, 3, GRID)


# Rejects returns_first_minimum and ties_to_smaller away from 1.0, in a shuffled list.
@case("an exact tie between 0.5 and 0.75, candidates in arbitrary order")
def tie_inside(ctx):
    # Squared errors are (123, 75, 75, 179, 279) / 4 for fractions 1, 0.75, 0.5, 0.25, 0.125.
    samples = np.array([[7.0, -5.0], [-9.0, 16.0]], dtype=np.float32)
    for order in ([0.5, 1.0, 0.75, 0.125, 0.25], [0.75, 0.125, 0.5, 0.25, 1.0]):
        got, _ = check_clip(ctx, samples, 2, np.array(order, np.float32), exact_ties=True)
        check.that(got == 0.75, "a tie must go to the larger candidate")
    # Mirrored samples tie the same way.
    check_clip(ctx, -samples, 2, np.array([0.5, 1.0, 0.75, 0.125, 0.25], np.float32),
               exact_ties=True)


@case("the winner moves toward the full range as bits increase")
def bits_sweep(ctx):
    samples = heavy_tailed(ctx, 16, 64, df=4.0)
    winners = [check_clip(ctx, samples, bits, GRID)[0] for bits in (2, 3, 5, 6, 8)]
    check.that(
        winners[0] < winners[-1],
        "8 bits must tolerate a wider range than 2 bits on the same samples",
        f"winners for bits 2, 3, 5, 6, 8: {winners}",
    )


@case("unsorted candidates, 1.0 not among them, and a single candidate")
def candidate_lists(ctx):
    samples = heavy_tailed(ctx, 16, 64)
    shuffled = ctx.rng.permutation(GRID[1:]).astype(np.float32)
    check_clip(ctx, samples, 4, shuffled)
    check_clip(ctx, samples, 4, np.array([0.9, 0.8, 0.95, 1.0, 0.85], np.float32))
    check_clip(ctx, samples, 4, np.array([0.6], np.float32), exact_ties=True)


@case("a flat list of samples, shape (N,)")
def flat_samples(ctx):
    check_clip(ctx, heavy_tailed(ctx, 1024), 5, GRID)
    check_clip(ctx, ctx.randn(1024), 3, GRID)


@case("the chosen range is never worse than no clipping on the calibration set")
def never_worse_than_no_clipping(ctx):
    def prop(rng):
        bits = int(rng.integers(2, 9))
        samples = heavy_tailed(ctx, 16, 64, df=float(rng.uniform(2.5, 8.0)))
        samples *= np.float32(10.0 ** rng.uniform(-2, 1))
        candidates = separated(samples, bits, GRID)
        got = float(ctx.call("best_clip", samples, bits, candidates))
        check.that(
            bool(np.isin(np.float32(got), candidates)),
            "best_clip must return one of the candidates",
            f"candidates = {candidates}, got {got}",
        )
        mse_got, mse_full = mse_table(samples, bits, [got, 1.0])
        check.that(
            mse_got <= mse_full * (1 + 1e-6),
            "the returned range has a higher MSE than no clipping at all",
            f"bits = {bits}: MSE at {got:g} is {mse_got:.6g}, MSE at 1.0 is {mse_full:.6g}",
        )
        ctx.check("best_clip", samples, bits, candidates)

    check.forall(prop, trials=8)
