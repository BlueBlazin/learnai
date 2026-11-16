"""Tests for temperature-scaling.

Numeric cases compare the solution with the oracle. Two property cases state what
temperature means without reference to an implementation: it never reorders the tokens,
and the entropy of the resulting distribution grows with it.
"""

from itertools import pairwise

import numpy as np

from learnai.judge import case, check

FN = "apply_temperature"
INF = np.inf


def temps(*values):
    return np.array(values, dtype=np.float32)


def entropy(row):
    """Entropy (nats) of softmax(row), in float64; -inf entries have probability 0."""
    x = np.asarray(row, dtype=np.float64)
    x = x[np.isfinite(x)]
    logp = x - x.max()
    logp = logp - np.log(np.exp(logp).sum())
    return float(-(np.exp(logp) * logp).sum())


@case("the worked example: temperatures 0.5, 2.0, and 0 (greedy, with a tie)", public=True)
def worked_example(ctx):
    logits = np.array(
        [[2.0, 1.0, 0.0, -INF], [2.0, 1.0, 0.0, -INF], [1.0, 3.0, 3.0, -1.0]], dtype=np.float32
    )
    got = ctx.call(FN, logits, temps(0.5, 2.0, 0.0))
    expected = np.array(
        [[4.0, 2.0, 0.0, -INF], [1.0, 0.5, 0.0, -INF], [-INF, 0.0, -INF, -INF]], dtype=np.float32
    )
    check.allclose(got, expected, label=FN, dtype=True)


@case("a random batch with a different positive temperature in every row", public=True)
def random_batch(ctx):
    # Catches one temperature used for the whole batch.
    logits = ctx.randn(4, 10, scale=3.0)
    ctx.check(FN, logits, temps(0.7, 1.0, 1.5, 0.2))


@case("temperature 0 is greedy: mixed with positive temperatures in one batch", public=True)
def greedy_rows(ctx):
    # Catches a division by zero, a temperature clamped to an epsilon, and a greedy row
    # that keeps the original logit instead of 0.
    logits = ctx.randn(5, 8, scale=2.0)
    logits[3] -= 20.0  # a greedy row whose largest logit is negative
    ctx.check(FN, logits, temps(0.0, 0.8, 0.0, 0.0, 1.3))


@case("greedy rows with ties: only the lowest index among the largest logits survives", public=True)
def greedy_ties(ctx):
    # Catches a greedy row built from `logits == max`, which keeps every tied entry.
    logits = np.array(
        [
            [0.5, 2.0, -1.0, 2.0, 2.0, 0.0],
            [3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
            [-4.0, -7.0, -4.0, -9.0, -5.0, -4.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    ctx.check(FN, logits, temps(0.0, 0.0, 0.0, 0.0))
    ctx.check(FN, logits, temps(0.0, 1.0, 0.0, 2.5))


@case("masked logits (-inf) stay -inf, in scaled rows and in greedy rows")
def masked_logits(ctx):
    # Catches arithmetic that turns -inf into NaN (-inf * 0, -inf - -inf). The greedy
    # rows have masked entries before and after their largest logit.
    logits = ctx.randn(4, 9, scale=2.0)
    logits[0, 0] = logits[0, 4] = -INF
    logits[1, 2] = logits[1, 8] = -INF
    logits[2, 1:] = -INF  # one unmasked token
    logits[3, :5] = -INF
    for t in (temps(0.6, 0.0, 0.0, 0.0), temps(0.0, 2.0, 0.3, 1.0), temps(1.0, 1.0, 1.0, 1.0)):
        got = ctx.check(FN, logits, t)
        check.that(not np.isnan(got).any(), f"{FN}: the result contains NaN")


@case("a square batch (B = V): the temperatures apply to rows, not to tokens")
def square_batch(ctx):
    # Temperatures broadcast along the vocabulary axis run without an error only when
    # B == V; this is where that mistake gives wrong numbers instead of an exception.
    logits = ctx.randn(6, 6, scale=2.0)
    ctx.check(FN, logits, temps(0.5, 1.0, 2.0, 4.0, 0.25, 3.0))
    ctx.check(FN, logits, temps(0.5, 0.0, 2.0, 0.0, 0.25, 0.0))


@case("temperature 1 returns the logits unchanged; all rows greedy; all rows equal")
def uniform_temperatures(ctx):
    logits = ctx.randn(4, 10, scale=3.0)
    logits[1, 3] = -INF
    ctx.check(FN, logits, temps(1.0, 1.0, 1.0, 1.0))
    ctx.check(FN, logits, temps(0.0, 0.0, 0.0, 0.0))
    ctx.check(FN, logits, temps(2.0, 2.0, 2.0, 2.0))


@case("a single row, a single token, and both at once")
def small_shapes(ctx):
    ctx.check(FN, ctx.randn(1, 12, scale=2.0), temps(0.4))
    ctx.check(FN, ctx.randn(1, 12, scale=2.0), temps(0.0))
    ctx.check(FN, np.array([[-3.5], [2.0], [0.0]], dtype=np.float32), temps(0.0, 0.5, 0.0))
    ctx.check(FN, np.array([[7.0]], dtype=np.float32), temps(0.0))


@case("temperatures from 0.05 to 100 and logits of magnitude 1e4")
def extreme_values(ctx):
    logits = ctx.randn(6, 10) * np.array(
        [[1e4], [1e3], [100.0], [1.0], [1e4], [0.01]], dtype=np.float32
    )
    ctx.check(FN, logits, temps(0.05, 100.0, 0.05, 37.5, 0.0, 0.0))


@case("a larger batch: B = 64, V = 1000, every fourth row greedy")
def larger_batch(ctx):
    logits = ctx.randn(64, 1000, scale=4.0)
    t = ctx.uniform(64, low=0.1, high=3.0)
    t[::4] = 0.0
    ctx.check(FN, logits, t)


@case("temperature never reorders tokens: scaled logits are sorted the same way as the input")
def ordering_preserved(ctx):
    shapes = [(4, 10), (6, 6), (2, 30)]  # few distinct shapes, for JAX

    def prop(rng):
        shape = shapes[int(rng.integers(len(shapes)))]
        logits = ctx.randn(*shape, scale=float(10.0 ** rng.uniform(-1, 2)))
        logits[rng.random(shape) < 0.1] = -INF
        # One entry per row is set after masking, so every row keeps a finite logit.
        logits[np.arange(shape[0]), rng.integers(shape[1], size=shape[0])] = 0.25
        t = (10.0 ** rng.uniform(-1, 1.5, size=shape[0])).astype(np.float32)
        t[rng.random(shape[0]) < 0.3] = 0.0
        out = ctx.call(FN, logits, t)
        check.shape(out, shape, label=FN)
        check.that(not np.isnan(out).any(), f"{FN}: the result contains NaN")
        for row, (x, o, temperature) in enumerate(zip(logits, out, t, strict=True)):
            where = f"row {row}, temperature = {temperature}\nlogits = {x}\nresult = {o}"
            top = int(np.argmax(x))  # numpy returns the first maximum
            check.that(
                int(np.argmax(o)) == top,
                f"{FN}: the most likely token of a row changed",
                where,
            )
            if temperature == 0:
                check.that(
                    np.isfinite(o).sum() == 1,
                    f"{FN}: a temperature-0 row must have exactly one finite entry",
                    where,
                )
                continue
            check.that(
                np.array_equal(np.isneginf(o), np.isneginf(x)),
                f"{FN}: the masked (-inf) entries of a row changed",
                where,
            )
            ranked = o[np.argsort(x, kind="stable")]
            ranked = ranked[np.isfinite(ranked)]
            check.that(
                bool((np.diff(ranked) >= 0).all()),
                f"{FN}: a larger logit got a smaller scaled logit than a smaller one",
                where,
            )

    check.forall(prop, trials=8)


@case("entropy grows with temperature: one row of logits at six increasing temperatures")
def entropy_monotone(ctx):
    # The same logits in every row, temperatures 0 < t1 < ... < t5. The entropy of
    # softmax(result) must be 0 for the greedy row and strictly increasing after it.
    # Catches multiplying by the temperature, which makes the entropy fall instead.
    def prop(rng):
        row = ctx.randn(12, scale=float(rng.uniform(1.0, 4.0)))
        row[int(rng.integers(12))] = -INF
        start, ratio = rng.uniform(0.1, 0.3), rng.uniform(1.6, 2.4)
        t = np.concatenate([[0.0], start * ratio ** np.arange(5)]).astype(np.float32)
        out = ctx.call(FN, np.tile(row, (6, 1)), t)
        check.shape(out, (6, 12), label=FN)
        check.that(not np.isnan(out).any(), f"{FN}: the result contains NaN")
        check.that(
            bool(np.isfinite(out).any(axis=-1).all()), f"{FN}: a row has no finite entry left"
        )
        entropies = [entropy(o) for o in out]
        where = f"logits (every row) = {row}\ntemperatures = {t}\nentropies = {entropies}"
        check.that(
            entropies[0] == 0.0,
            f"{FN}: a temperature-0 row must give a one-hot distribution (entropy 0)",
            where,
        )
        check.that(
            all(b > a + 1e-6 for a, b in pairwise(entropies)),
            f"{FN}: the entropy of softmax(result) does not increase with the temperature",
            where,
        )

    check.forall(prop, trials=6)
