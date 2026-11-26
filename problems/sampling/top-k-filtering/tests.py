"""Tests for top-k-filtering.

Numeric cases compare the solution with the oracle. The filter only compares and copies
input values, so there is no rounding to allow for: kept entries must equal the input
and removed entries must be -inf. The property case states the definition (exactly
min(k, finite) survivors, the largest ones, ties to the lowest index) in numpy.
"""

import numpy as np

from learnai.judge import case, check

FN = "top_k_filter"
INF = np.inf


def quantized(ctx, *shape, levels=4):
    """Logits drawn from a few distinct values, so that ties are everywhere."""
    return (ctx.randint(0, levels, shape) * 0.5 - 1.0).astype(np.float32)


@case("the worked example: k = 2, a tie at the boundary, and a masked entry", public=True)
def worked_example(ctx):
    logits = np.array([[1.0, 3.0, 0.5, 2.0, -1.0], [0.0, 2.0, -INF, 2.0, 2.0]], dtype=np.float32)
    got = ctx.call(FN, logits, 2)
    expected = np.array(
        [[-INF, 3.0, -INF, 2.0, -INF], [-INF, 2.0, -INF, 2.0, -INF]], dtype=np.float32
    )
    check.allclose(got, expected, label=FN, dtype=True)


@case("a random batch: k = 3 of 10 tokens, then k = 1 and k = 9", public=True)
def random_batch(ctx):
    # Catches one ranking shared by the whole batch, and keeping k + 1 entries.
    logits = ctx.randn(4, 10, scale=3.0) + np.array([[0.0], [5.0], [-5.0], [10.0]], np.float32)
    for k in (3, 1, 9):
        ctx.check(FN, logits, k)


@case("ties at the boundary: equal logits are ranked by token index, lowest first", public=True)
def boundary_ties(ctx):
    # Catches `logits >= threshold` (keeps every tied entry) and a ranking that puts
    # the highest index first among equals.
    logits = np.array(
        [
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [0.0, 2.0, 2.0, 5.0, 2.0, 2.0],
            [3.0, -1.0, 3.0, -1.0, 3.0, -1.0],
            [4.0, 4.0, 0.0, 0.0, 0.0, 9.0],
        ],
        dtype=np.float32,
    )
    for k in (1, 2, 3, 4, 5):
        ctx.check(FN, logits, k)


@case("k = 0 (top-k disabled) and k >= V return the logits unchanged", public=True)
def no_op(ctx):
    # Catches a missing k = 0 special case (everything masked) and code that indexes
    # the k-th largest value when k > V.
    logits = ctx.randn(3, 7, scale=2.0)
    logits[1, 2] = -INF
    for k in (0, 7, 8, 1000):
        ctx.check(FN, logits, k)


@case("masked logits (-inf) stay masked, also when a row has fewer than k finite entries")
def masked_logits(ctx):
    # Catches a fill value other than -inf: the removed entries must be exactly -inf.
    logits = ctx.randn(5, 8, scale=2.0)
    logits[0, 1] = logits[0, 6] = -INF
    logits[1, :6] = -INF  # two finite entries
    logits[2, 1:] = -INF  # one finite entry
    logits[3, 3:] = -INF  # three finite entries, all tied
    logits[3, :3] = 0.75
    for k in (1, 2, 3, 5, 7):
        got = ctx.check(FN, logits, k)
        finite = np.isfinite(got).sum(axis=-1)
        want = np.minimum(k, np.isfinite(logits).sum(axis=-1))
        check.that(
            np.array_equal(finite, want),
            f"{FN}: got {finite.tolist()} finite entries per row, expected {want.tolist()} "
            "(rows with masked logits)",
            f"k = {k}\nlogits = {logits}\nresult = {got}",
        )


@case("a square batch (B = V) and k = 1 with tied maxima")
def square_and_greedy(ctx):
    square = ctx.randn(6, 6, scale=2.0)
    for k in (1, 2, 5):
        ctx.check(FN, square, k)
    tied = quantized(ctx, 6, 6, levels=2)
    for k in (1, 3):
        ctx.check(FN, tied, k)


@case("quantized logits: ties everywhere, every k from 1 to V - 1")
def many_ties(ctx):
    logits = quantized(ctx, 4, 10, levels=3)
    logits[2, 4] = logits[3, 0] = -INF
    for k in range(1, 10):
        ctx.check(FN, logits, k)


@case("a single row, a single token, and logits of magnitude 1e4")
def small_and_large(ctx):
    single = ctx.randn(1, 12, scale=2.0)
    for k in (0, 4, 12):
        ctx.check(FN, single, k)
    for k in (0, 1, 2):
        ctx.check(FN, np.array([[3.5], [-200.0]], dtype=np.float32), k)
    big = ctx.randn(4, 10) * np.array([[1e4], [1e-3], [50.0], [1.0]], dtype=np.float32)
    ctx.check(FN, big, 4)


@case("a vocabulary of 2000 tokens with k = 40, with and without ties")
def large_vocabulary(ctx):
    ctx.check(FN, ctx.randn(2, 2000, scale=3.0), 40)
    ctx.check(FN, quantized(ctx, 2, 2000, levels=60), 40)


@case("the definition: min(k, finite) survivors, the largest logits, ties to the lowest index")
def top_k_properties(ctx):
    shapes = [(4, 10), (6, 6), (4, 6)]  # shapes other cases already use, for JAX

    def prop(rng):
        shape = shapes[int(rng.integers(len(shapes)))]
        if rng.random() < 0.5:
            logits = quantized(ctx, *shape, levels=int(rng.integers(2, 6)))
        else:
            logits = ctx.randn(*shape, scale=float(10.0 ** rng.uniform(-1, 2)))
        logits[rng.random(shape) < 0.15] = -INF
        # One entry per row is set after masking, so every row keeps a finite logit.
        logits[np.arange(shape[0]), rng.integers(shape[1], size=shape[0])] = 0.5
        k = int(rng.integers(0, shape[1] + 3))
        out = ctx.call(FN, logits, k)
        check.shape(out, shape, label=FN)
        check.that(not np.isnan(out).any(), f"{FN}: the result contains NaN")
        limit = shape[1] if k == 0 else k
        for row, (x, o) in enumerate(zip(logits, out, strict=True)):
            where = f"row {row}, k = {k}\nlogits = {x}\nresult = {o}"
            kept = np.isfinite(o)
            check.that(
                bool(((o == x) | np.isneginf(o)).all()),
                f"{FN}: an entry is neither its input logit nor -inf",
                where,
            )
            survivors = min(limit, np.isfinite(x).sum())
            check.that(
                kept.sum() == survivors,
                f"{FN}: a row has {kept.sum()} survivors, expected {survivors} (min(k, number of "
                "finite logits))",
                where,
            )
            # Rank by decreasing logit, then by increasing index: the survivors must be
            # a prefix of that ranking.
            ranking = np.lexsort((np.arange(len(x)), -x))
            check.that(
                bool(kept[ranking[: kept.sum()]].all()),
                f"{FN}: the survivors are not the largest logits (ties: lowest index first)",
                where,
            )

    check.forall(prop, trials=10)
