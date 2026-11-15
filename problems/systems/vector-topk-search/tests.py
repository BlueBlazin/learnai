"""Tests for vector-topk-search.

Scores are compared with the oracle to the tolerance of the statement. Indices are
compared exactly, with one allowance in the random cases: where the solution and the
oracle disagree about two vectors whose true scores (a float64 scan by the kit) are
closer than float32 can tell apart, either order is accepted. The tie cases use
vectors whose scores are exact in float32 whatever the order of the arithmetic
(small integers, or unit vectors with entries of 0 and +-0.5), so there the tie rule
is checked strictly.
"""

import numpy as np

from whetstone.judge import case, check
from whetstone.kit import vectors

FN = "search"
TOL = {"rtol": 1e-4, "atol": 1e-5}  # the tolerance stated in Constraints


def check_search(ctx, queries, corpus, k, metric="cosine", *, strict=False):
    queries = np.asarray(queries, dtype=np.float32)
    corpus = np.asarray(corpus, dtype=np.float32)
    label = f"search(k={k}, metric={metric!r})"
    got = ctx.call(FN, queries, corpus, k, metric=metric)
    check.that(
        isinstance(got, tuple) and len(got) == 2,
        f"{label}: must return a tuple (indices, scores)",
        f"got: {type(got).__name__}",
    )
    indices, scores = got
    want_indices, want_scores = ctx.expected(FN, queries, corpus, k, metric=metric)
    check.shape(indices, want_indices.shape, label=f"{label}: indices")
    check.that(
        indices.dtype.kind in "iu",
        f"{label}: indices have dtype {indices.dtype}; they must be integers",
    )
    check.allclose(scores, want_scores, **TOL, dtype=True, label=f"{label}: scores")
    if np.array_equal(indices, want_indices):
        return
    if not strict:
        # A swap between two vectors that float32 cannot tell apart is not an error.
        table = vectors.scores(queries, corpus, metric)
        rows, cols = np.nonzero(indices != want_indices)
        valid = (indices >= 0).all() and (indices < corpus.shape[0]).all()
        if valid:
            got_true = table[rows, indices[rows, cols]]
            want_true = table[rows, want_indices[rows, cols]]
            distinct = all(len(set(row.tolist())) == len(row) for row in indices)
            if distinct and np.all(np.abs(got_true - want_true) <= 1e-6):
                return
    check.equal(indices, want_indices, label=f"{label}: indices")


def corpus_and_queries(rng, n, dim, q, *, clusters=8):
    """A clustered corpus whose vectors have very different lengths, and queries
    near some of its vectors (also of different lengths)."""
    data = vectors.clustered(rng, n=n, dim=dim, clusters=clusters, spread=0.6)
    corpus = data.vectors * rng.uniform(0.2, 5.0, size=(n, 1)).astype(np.float32)
    near = vectors.queries_near(rng, data.vectors, rng.integers(0, n, size=q), noise=0.3)
    queries = near * rng.uniform(0.5, 3.0, size=(q, 1)).astype(np.float32)
    return corpus, queries


EXAMPLE_CORPUS = [[1.0, 0.0], [0.0, 2.0], [3.0, 3.0], [2.0, 0.0], [-1.0, 0.0]]
EXAMPLE_QUERIES = [[1.0, 0.0], [1.0, 1.0]]


@case("the worked example: 2 queries, 5 vectors, k = 2, both metrics", public=True)
def worked_example(ctx):
    queries = np.array(EXAMPLE_QUERIES, dtype=np.float32)
    corpus = np.array(EXAMPLE_CORPUS, dtype=np.float32)
    indices, scores = ctx.call(FN, queries, corpus, 2)
    check.equal(indices, np.array([[0, 3], [2, 0]]), label="search(k=2): indices")
    check.allclose(scores, [[1.0, 1.0], [1.0, 0.70710678]], **TOL, label="search(k=2): scores")
    indices, scores = ctx.call(FN, queries, corpus, 2, metric="dot")
    check.equal(indices, np.array([[2, 3], [2, 1]]), label="search(k=2, 'dot'): indices")
    check.allclose(scores, [[3.0, 2.0], [6.0, 2.0]], **TOL, label="search(k=2, 'dot'): scores")
    check_search(ctx, queries, corpus, 2, strict=True)
    check_search(ctx, queries, corpus, 2, "dot", strict=True)


# Rejects unnormalized_dot and normalizes_corpus_only: the vectors differ in length.
@case("cosine: 2,000 vectors of very different lengths, 8 queries, k = 10", public=True)
def cosine_random(ctx):
    corpus, queries = corpus_and_queries(ctx.rng, 2000, 32, 8)
    check_search(ctx, queries, corpus, 10)


# Rejects ignores_metric.
@case("dot: the same kind of corpus ranked by inner product", public=True)
def dot_random(ctx):
    corpus, queries = corpus_and_queries(ctx.rng, 2000, 32, 8)
    check_search(ctx, queries, corpus, 10, "dot")


def integer_vectors(rng, n, dim, low, high):
    return rng.integers(low, high + 1, size=(n, dim)).astype(np.float32)


# Rejects topk_arbitrary_ties and unstable_full_sort: the scores are small integers,
# so most of the top 10 are ties, also across the boundary between rank 10 and 11.
@case("ties: integer vectors under 'dot', equal scores ordered by the lower index",
      public=True)  # fmt: skip
def ties_dot(ctx):
    rng = ctx.rng
    corpus = integer_vectors(rng, 64, 8, -1, 1)
    queries = integer_vectors(rng, 4, 8, -1, 2)
    check_search(ctx, queries, corpus, 10, "dot", strict=True)
    check_search(ctx, queries, corpus, 3, "dot", strict=True)


# Rejects loop_over_corpus. Measured on an M1 Max, as multiples of the reference's time:
# one product per corpus vector, 90x (torch); a stable full sort of every row, 1.7x
# (torch) and 4.2x (JAX); a loop over the four queries with a full sort each, 5x.
@case("speed: within 10x of the reference for 4 queries against 20,000 vectors, k = 10",
      public=True)  # fmt: skip
def fast_enough(ctx):
    rng = ctx.rng
    corpus = rng.standard_normal((20000, 64)).astype(np.float32)
    queries = rng.standard_normal((4, 64)).astype(np.float32)
    ctx.check_speed(FN, queries, corpus, 10, max_ratio=10.0, rounds=5)


def half_unit_vectors(rng, n, dim=8):
    """Vectors with exactly four entries of +-1: length 2, so the normalized entries
    are 0 and +-0.5 and every cosine is an exact multiple of 0.25."""
    out = np.zeros((n, dim), dtype=np.float32)
    for row in out:
        row[rng.choice(dim, size=4, replace=False)] = rng.choice([-1.0, 1.0], size=4)
    return out


# Rejects topk_arbitrary_ties and unstable_full_sort under the cosine metric.
@case("ties: duplicate vectors and exactly equal cosines")
def ties_cosine(ctx):
    rng = ctx.rng
    corpus = half_unit_vectors(rng, 64)
    corpus[40:52] = corpus[3:15]  # twelve exact duplicates, far from their originals
    queries = half_unit_vectors(rng, 4)
    queries[0] = corpus[5]
    check_search(ctx, queries, corpus, 10, strict=True)
    check_search(ctx, queries * 3.0, corpus, 3, strict=True)


# Rejects k_exceeds_n.
@case("k equal to the corpus size, and larger than it: min(k, N) columns")
def k_at_least_n(ctx):
    rng = ctx.rng
    corpus = integer_vectors(rng, 64, 8, -1, 1)
    queries = integer_vectors(rng, 4, 8, -1, 2)
    check_search(ctx, queries, corpus, 64, "dot", strict=True)
    check_search(ctx, queries, corpus, 100, "dot", strict=True)
    corpus, queries = corpus_and_queries(rng, 6, 32, 8, clusters=2)
    check_search(ctx, queries, corpus, 10)
    check_search(ctx, queries, corpus[:1], 10)


@case("k = 1, and a single query")
def smallest_sizes(ctx):
    rng = ctx.rng
    corpus, queries = corpus_and_queries(rng, 2000, 32, 8)
    check_search(ctx, queries, corpus, 1)
    check_search(ctx, queries, corpus, 1, "dot")
    check_search(ctx, queries[:1], corpus, 10)


@case("all scores of a row equal: the first k indices")
def constant_rows(ctx):
    corpus = np.tile(np.array([[1.0, -1.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0]], np.float32), (64, 1))
    queries = half_unit_vectors(ctx.rng, 4)
    check_search(ctx, queries, corpus, 10, strict=True)
    check_search(ctx, queries, corpus, 10, "dot", strict=True)


@case("random corpora: both metrics, several k")
def random_sweep(ctx):
    rng = ctx.rng
    for trial in range(4):
        corpus, queries = corpus_and_queries(rng, 2000, 32, 8, clusters=int(rng.integers(2, 20)))
        k = (1, 10, 3, 10)[trial]
        check_search(ctx, queries, corpus, k, ("cosine", "dot")[trial % 2])
