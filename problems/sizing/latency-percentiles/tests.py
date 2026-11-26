"""Tests for latency-percentiles.

Worked examples are compared with literal values, the rest with the oracle, with
numpy.percentile as an independent check of the linear method, and with property cases
that state what a percentile means without reference to the oracle.
"""

import math
from fractions import Fraction

import numpy as np

from learnai.judge import case, check

LATENCIES = [120.0, 95.0, 110.0, 2300.0, 105.0, 98.0, 130.0, 101.0, 99.0, 115.0]
SHARD_A = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0]
SHARD_B = [100.0, 900.0]
COMMON_Q = (0, 1, 25, 50, 75, 90, 95, 99, 100)


def exact_rank(q, n):
    """The nearest rank computed without floating point, reading q as a decimal."""
    return max(1, math.ceil(Fraction(str(q)) * n / 100))


def latencies(rng, n):
    """Skewed latencies in milliseconds, rounded so that duplicates occur."""
    return [round(float(v), 1) for v in rng.lognormal(mean=4.0, sigma=0.8, size=n)]


def check_nearest(ctx, values, q, label=None, expected=None):
    label = label or f"percentile_nearest_rank of {len(values)} values, q={q}"
    # The statement promises inputs for which the multiply-first formula is exact.
    assert math.ceil(q * len(values) / 100) == math.ceil(Fraction(str(q)) * len(values) / 100)
    got = ctx.call("percentile_nearest_rank", list(values), q)
    if expected is None:
        expected = sorted(values)[exact_rank(q, len(values)) - 1]
    check.that(
        isinstance(got, float) and got == expected,
        f"{label}: got {got!r}, expected {expected!r}",
        f"values (first 20) = {values[:20]!r}\nq = {q!r}\nexpected: {expected!r}\n     got: {got!r}",
    )


def check_linear(ctx, values, q, label=None, expected=None):
    label = label or f"percentile_linear of {len(values)} values, q={q}"
    got = ctx.call("percentile_linear", list(values), q)
    check.that(isinstance(got, float), f"{label}: the result must be a float", f"got: {got!r}")
    if expected is None:
        expected = ctx.expected("percentile_linear", list(values), q)
    check.allclose(got, expected, rtol=1e-9, atol=1e-9, label=label)


def check_merge(ctx, shards, q, label=None, expected=None):
    label = label or f"merge_percentile of {len(shards)} shards, q={q}"
    pooled = [v for shard in shards for v in shard]
    assert math.ceil(q * len(pooled) / 100) == math.ceil(Fraction(str(q)) * len(pooled) / 100)
    got = ctx.call("merge_percentile", [list(shard) for shard in shards], q)
    if expected is None:
        expected = sorted(pooled)[exact_rank(q, len(pooled)) - 1]
    check.that(
        isinstance(got, float) and got == expected,
        f"{label}: got {got!r}, expected {expected!r}",
        f"shard sizes = {[len(s) for s in shards]!r}\nq = {q!r}\n"
        f"expected: {expected!r}\n     got: {got!r}",
    )


@case("the worked example: ten latencies in arrival order", public=True)
def worked_example(ctx):
    # Rejects assumes_sorted, rank_floor (p99 of ten), zero_based_rank,
    # linear_is_nearest_rank, and linear_position_uses_n.
    for q, expected in ((50, 105.0), (90, 130.0), (99, 2300.0), (0, 95.0)):
        check_nearest(ctx, LATENCIES, q, f"percentile_nearest_rank(latencies, {q})", expected)
    for q, expected in ((50, 107.5), (90, 347.0), (99, 2104.7)):
        check_linear(ctx, LATENCIES, q, f"percentile_linear(latencies, {q})", expected)


@case("the two-server example: the pooled p90 is not a mean of shard p90s", public=True)
def two_servers(ctx):
    # Rejects average_shard_percentiles (458.5) and weighted_average_shards (193.6).
    check_nearest(ctx, SHARD_A, 90, "percentile_nearest_rank(a, 90)", 17.0)
    check_nearest(ctx, SHARD_B, 90, "percentile_nearest_rank(b, 90)", 900.0)
    check_merge(ctx, [SHARD_A, SHARD_B], 90, "merge_percentile([a, b], 90)", 100.0)
    check_merge(ctx, [SHARD_B, SHARD_A], 90, "merge_percentile([b, a], 90)", 100.0)
    check_merge(ctx, [SHARD_A, SHARD_B], 50, "merge_percentile([a, b], 50)", 14.0)


@case("ranks that q / 100 * n gets wrong: p99.9 of 1000, p55 of 100, p56 of 25", public=True)
def exact_ranks(ctx):
    # Rejects divide_first: in each pair q * n / 100 is a whole number, and q / 100 * n
    # evaluates to just above it.
    for q, n in ((99.9, 1000), (55, 100), (28, 100), (7, 100), (56, 25), (14, 50)):
        values = [float(v) for v in ctx.rng.permutation(n)]
        check_nearest(ctx, values, q, f"percentile_nearest_rank of a shuffled 0..{n - 1}, q={q}")
        check_merge(
            ctx,
            [values[: n // 3], values[n // 3 :]],
            q,
            f"merge_percentile of a shuffled 0..{n - 1} in two shards, q={q}",
        )


@case("q = 0 and q = 100, a single value, two values, all values equal", public=True)
def edges(ctx):
    for values in ([42.5], [7.0, 3.0], [5.0, 5.0, 5.0, 5.0], LATENCIES):
        for q in (0, 100, 50, 99.99, 0.01):
            check_nearest(ctx, values, q, f"percentile_nearest_rank({values!r}, {q})")
            check_linear(ctx, values, q, f"percentile_linear({values!r}, {q})")
    check_merge(ctx, [[42.5]], 99, "merge_percentile([[42.5]], 99)")
    check_merge(ctx, [[], [3.0], []], 0, "merge_percentile([[], [3.0], []], 0)")


@case("the caller's lists are not modified")
def inputs_unchanged(ctx):
    # Rejects sorts_in_place. The solution is called directly: ctx.call passes a copy.
    values = list(LATENCIES)
    ctx.solution.percentile_nearest_rank(values, 90)
    check.that(values == LATENCIES, "percentile_nearest_rank modified its input list")
    ctx.solution.percentile_linear(values, 90)
    check.that(values == LATENCIES, "percentile_linear modified its input list")
    shards = [list(SHARD_B), list(SHARD_A), list(LATENCIES)]
    ctx.solution.merge_percentile(shards, 90)
    check.that(shards == [SHARD_B, SHARD_A, LATENCIES], "merge_percentile modified its input lists")


@case("nearest rank on skewed latencies: sample sizes 1 to 1000, common percentiles")
def nearest_rank_sweep(ctx):
    for n in (1, 2, 3, 9, 10, 11, 99, 100, 101, 200, 1000):
        values = latencies(ctx.rng, n)
        for q in COMMON_Q:
            check_nearest(ctx, values, q)
    values = latencies(ctx.rng, 2500)
    for q in (99.9, 99.99, 0.1, 97.5, 2.5):
        check_nearest(ctx, values, q)


@case("linear interpolation agrees with numpy.percentile")
def linear_matches_numpy(ctx):
    for n in (1, 2, 3, 10, 11, 100, 101, 1000):
        values = latencies(ctx.rng, n)
        for q in (*COMMON_Q, 99.9, 33.3, 12.5):
            label = f"percentile_linear of {n} values, q={q}"
            check_linear(ctx, values, q, label, float(np.percentile(values, q)))


@case("four shards of very different sizes and speeds: pooled p50, p90, p99, p99.9")
def uneven_shards(ctx):
    # The fixture the curriculum asks for: one small, slow shard next to large fast ones.
    rng = ctx.rng
    shards = [
        [round(float(v), 1) for v in rng.lognormal(3.5, 0.3, size=4000)],
        [round(float(v), 1) for v in rng.lognormal(3.6, 0.4, size=2500)],
        [round(float(v), 1) for v in rng.lognormal(6.5, 0.9, size=60)],
        [],
        [round(float(v), 1) for v in rng.lognormal(3.4, 0.3, size=940)],
    ]
    pooled = [v for shard in shards for v in shard]
    for q in (50, 90, 99, 99.9):
        pooled_value = sorted(pooled)[exact_rank(q, len(pooled)) - 1]
        per_shard = [sorted(s)[exact_rank(q, len(s)) - 1] for s in shards if s]
        mean = sum(per_shard) / len(per_shard)
        weighted = sum(len(s) * sorted(s)[exact_rank(q, len(s)) - 1] for s in shards if s)
        # The fixture must separate pooling from both ways of averaging.
        assert abs(mean - pooled_value) > 1.0, (q, mean, pooled_value)
        assert abs(weighted / len(pooled) - pooled_value) > 0.5, (q, weighted, pooled_value)
        check_merge(ctx, shards, q)
        check_merge(ctx, shards[::-1], q, f"merge_percentile of the shards reversed, q={q}")


@case("property: the nearest-rank percentile is the smallest observation covering q percent")
def nearest_rank_property(ctx):
    def prop(rng):
        n = int(rng.integers(1, 400))
        values = latencies(rng, n)
        q = int(rng.integers(0, 101))
        got = ctx.call("percentile_nearest_rank", list(values), q)
        detail = f"n = {n}, q = {q}\nvalues (first 20) = {values[:20]!r}\ngot = {got!r}"
        check.that(got in values, "the result is not one of the observations", detail)
        at_or_below = sum(v <= got for v in values)
        below = sum(v < got for v in values)
        check.that(
            100 * at_or_below >= q * n,
            "fewer than q percent of the values are <= the result",
            detail + f"\nvalues <= result: {at_or_below} of {n}",
        )
        check.that(
            100 * below < q * n or below == 0,
            "a smaller observation already has q percent of the values at or below it",
            detail + f"\nvalues < result: {below} of {n}",
        )

    check.forall(prop, trials=40)


@case("property: the linear percentile lies between the observations around its position")
def linear_property(ctx):
    def prop(rng):
        n = int(rng.integers(1, 300))
        values = latencies(rng, n)
        q = float(rng.integers(0, 1001)) / 10
        got = ctx.call("percentile_linear", list(values), q)
        ordered = sorted(values)
        h = (n - 1) * q / 100
        lo = ordered[max(0, math.floor(h - 1e-9))]
        hi = ordered[min(n - 1, math.ceil(h + 1e-9))]
        check.that(
            lo - 1e-9 <= got <= hi + 1e-9,
            "the result is outside the two observations that surround position (n - 1) * q / 100",
            f"n = {n}, q = {q}\nsurrounding observations: {lo!r} and {hi!r}\ngot = {got!r}",
        )

    check.forall(prop, trials=40)


@case("property: merge_percentile does not depend on how samples are split into shards")
def merge_property(ctx):
    def prop(rng):
        n = int(rng.integers(1, 600))
        values = latencies(rng, n)
        q = int(rng.integers(0, 101))
        n_shards = int(rng.integers(1, 8))
        # Sort a random part of the data into the last shard so that shards differ in
        # speed as well as size, the situation where averaging goes wrong.
        ordered = sorted(values)
        cut = int(rng.integers(0, n + 1))
        fast, slow = ordered[:cut], ordered[cut:]
        owner = rng.integers(0, n_shards, size=len(fast))
        shards = [[v for v, o in zip(fast, owner, strict=True) if o == k] for k in range(n_shards)]
        shards.append(slow)
        got = ctx.call("merge_percentile", shards, q)
        whole = ctx.call("percentile_nearest_rank", list(values), q)
        check.that(
            got == whole,
            f"merge_percentile returned {got!r}, but percentile_nearest_rank of the pooled "
            f"samples returned {whole!r}",
            f"shard sizes = {[len(s) for s in shards]!r}, q = {q}\n"
            f"merge_percentile = {got!r}\npercentile_nearest_rank(pooled) = {whole!r}",
        )

    check.forall(prop, trials=40)
