"""Tests for ivf-index-search.

`build_ivf` is compared exactly with the oracle on corpora whose assignments are
unambiguous in float32 (vectors whose two nearest centroids are almost equally far
are removed), plus integer examples where ties are exact. `search` is graded on the
oracle's index, so a wrong build does not hide a right search; one case runs the
solution's own build and search end to end.

Search distances come from the kit's `CountingDistance`, which is also the cost
meter. The main corpus and its queries are module constants, so that the candidate
counts (and with them the shapes JAX compiles for) repeat across cases.
"""

from itertools import pairwise

import numpy as np

from whetstone.judge import case, check
from whetstone.kit import vectors

TOL = {"rtol": 1e-4, "atol": 1e-5}  # the tolerance stated in Constraints


def sq_dists(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)


def make_corpus(rng, n, dim, clusters, *, spread=0.9, scale=True):
    """(corpus, centroids): overlapping clusters, vectors of different lengths, and
    no vector whose two nearest centroids are nearly equally far."""
    data = vectors.clustered(rng, n=n, dim=dim, clusters=clusters, spread=spread, normalize=False)
    corpus = data.vectors
    if scale:
        corpus = corpus * rng.uniform(0.6, 1.6, size=(n, 1)).astype(np.float32)
    table = np.sort(sq_dists(corpus, data.centroids), axis=1)
    clear = (table[:, 1] - table[:, 0]) > 1e-3 if clusters > 1 else np.ones(n, dtype=bool)
    return np.ascontiguousarray(corpus[clear]), data.centroids


def truth_index(corpus, centroids):
    assigned = sq_dists(corpus, centroids).argmin(axis=1)
    ids = np.argsort(assigned, kind="stable")
    sizes = np.bincount(assigned, minlength=len(centroids))
    return np.concatenate([[0], np.cumsum(sizes)]), ids


def check_build(ctx, corpus, centroids, label="build_ivf"):
    got = ctx.call("build_ivf", corpus, centroids)
    check.that(
        isinstance(got, tuple) and len(got) == 4,
        f"{label}: must return a tuple (centroids, offsets, ids, vectors)",
        f"got: {type(got).__name__}",
    )
    got_centroids, offsets, ids, stored = got
    want_offsets, want_ids = truth_index(corpus, centroids)
    check.allclose(got_centroids, centroids, label=f"{label}: centroids")
    check.shape(offsets, want_offsets.shape, label=f"{label}: offsets")
    check.equal(offsets, want_offsets, label=f"{label}: offsets")
    check.shape(ids, want_ids.shape, label=f"{label}: ids")
    if not np.array_equal(ids, want_ids):
        same_lists = all(
            sorted(ids[a:b].tolist()) == want_ids[a:b].tolist() for a, b in pairwise(want_offsets)
        )
        check.that(
            not same_lists,
            f"{label}: every list holds the right vectors, but not in ascending id order",
            f"expected ids =\n{want_ids}\ngot ids =\n{ids}",
        )
        check.equal(ids, want_ids, label=f"{label}: ids")
    check.allclose(stored, corpus[want_ids], label=f"{label}: vectors")
    return got


def oracle_index(ctx, corpus, centroids):
    index = ctx.expected("build_ivf", corpus, centroids)
    want_offsets, want_ids = truth_index(corpus, centroids)
    ok = np.array_equal(index[1], want_offsets) and np.array_equal(index[2], want_ids)
    check.that(ok, "test bug: the oracle's index is not the brute-force one")
    return index


def check_search(ctx, q, index, k, nprobe, *, metric=False, label=None):
    label = label or f"search(k={k}, nprobe={nprobe})"
    distance = vectors.CountingDistance(ctx.fw, metric="l2")
    got = ctx.call("search", q, index, k, nprobe, distance)
    check.that(
        isinstance(got, tuple) and len(got) == 2,
        f"{label}: must return a tuple (ids, distances)",
        f"got: {type(got).__name__}",
    )
    ids, dists = got
    want_ids, want_dists = ctx.expected(
        "search", q, index, k, nprobe, vectors.CountingDistance(ctx.oracle_fw, metric="l2")
    )
    check.shape(ids, want_ids.shape, label=f"{label}: ids")
    check.that(
        len(set(ids.tolist())) == len(ids),
        f"{label}: the result contains the same id more than once",
        f"got ids = {ids.tolist()}",
    )
    check.equal(ids, want_ids, label=f"{label}: ids")
    check.allclose(dists, want_dists, **TOL, dtype=True, label=f"{label}: distances")
    # The cost: one distance per centroid, one per vector of the probed lists.
    centroids, offsets = index[0], index[1]
    order = np.argsort(sq_dists(q[None], centroids)[0], kind="stable")[:nprobe]
    scanned = int(sum(offsets[j + 1] - offsets[j] for j in order))
    expected = len(centroids) + scanned
    if metric:
        check.cost(distance.counters, "distances", exactly=expected, unit="distances")
    else:
        computed = distance.counters.distances
        check.that(
            computed == expected,
            f"{label}: {computed} distances were computed with `distance`, which is not the "
            "number of centroids plus the sizes of the probed lists",
            f"expected {len(centroids)} + {scanned} = {expected}",
        )
    return ids


# The main corpus: about 2,000 vectors of 32 dimensions around 8 overlapping centroids.
_RNG = np.random.default_rng(60)
CORPUS, CENTROIDS = make_corpus(_RNG, 2000, 32, 8, spread=1.5)
# Two queries halfway between two centroids, where one probe is not enough, and one
# next to a corpus vector.
QUERIES = np.concatenate(
    [
        0.5 * (CENTROIDS[[0, 2]] + CENTROIDS[[1, 3]])
        + 0.1 * _RNG.standard_normal((2, 32)).astype(np.float32),
        vectors.queries_near(_RNG, CORPUS, [1500], noise=0.4, normalize=False),
    ]
).astype(np.float32)

EXAMPLE_CENTROIDS = np.array([[0, 0], [4, 0], [0, 4]], dtype=np.float32)
EXAMPLE_CORPUS = np.array([[1, 0], [4, 1], [0, 1], [2, 0], [5, 0], [3, -1]], dtype=np.float32)
EXAMPLE_Q = np.array([2.5, 0.0], dtype=np.float32)


@case("the worked example: 6 vectors, 3 lists (one empty), nprobe 1, 2, and 3", public=True)
def worked_example(ctx):
    index = check_build(ctx, EXAMPLE_CORPUS, EXAMPLE_CENTROIDS)
    check.equal(index[1], np.array([0, 3, 6, 6]), label="build_ivf: offsets")
    check.equal(index[2], np.array([0, 2, 3, 1, 4, 5]), label="build_ivf: ids")
    index = oracle_index(ctx, EXAMPLE_CORPUS, EXAMPLE_CENTROIDS)
    for (k, nprobe), (want_ids, want_dists) in {
        (2, 1): ([5, 1], [1.25, 3.25]),
        (2, 2): ([3, 5], [0.25, 1.25]),
        (2, 3): ([3, 5], [0.25, 1.25]),
        (10, 1): ([5, 1, 4], [1.25, 3.25, 6.25]),
    }.items():
        label = f"search(k={k}, nprobe={nprobe})"
        ids = check_search(ctx, EXAMPLE_Q, index, k, nprobe, metric=True, label=label)
        check.equal(ids, np.array(want_ids), label=f"{label}: ids")
        got = ctx.call("search", EXAMPLE_Q, index, k, nprobe,
                       vectors.CountingDistance(ctx.fw, metric="l2"))  # fmt: skip
        check.allclose(got[1], want_dists, **TOL, label=f"{label}: distances")


# Rejects assigns_by_inner_product and ids_descending_within_list.
@case("build_ivf: 2,000 vectors of different lengths into 8 lists", public=True)
def build_lists(ctx):
    check_build(ctx, CORPUS, CENTROIDS)


# Rejects probes_by_id, scans_everything, own_centroid_distances, returns_positions.
@case("search: the nearest lists are scanned, and only they are paid for", public=True)
def search_cost(ctx):
    index = oracle_index(ctx, CORPUS, CENTROIDS)
    check_search(ctx, QUERIES[0], index, 10, 1)
    check_search(ctx, QUERIES[1], index, 10, 2)
    check_search(ctx, QUERIES[2], index, 10, 2)
    check_search(ctx, QUERIES[0], index, 10, 2, metric=True)


@case("nprobe equal to the number of lists gives the exact nearest neighbors", public=True)
def exhaustive_is_exact(ctx):
    index = oracle_index(ctx, CORPUS, CENTROIDS)
    truth, _ = vectors.exact_topk(QUERIES, CORPUS, 10, metric="l2")
    for q, want in zip(QUERIES, truth, strict=True):
        ids = check_search(ctx, q, index, 10, 8)
        check.equal(ids, want, label="search(k=10, nprobe=8): ids against an exact scan")


@case("recall@10 never falls as nprobe grows, and reaches 1 at nprobe = 8", public=True)
def recall_is_monotone(ctx):
    index = oracle_index(ctx, CORPUS, CENTROIDS)
    truth, _ = vectors.exact_topk(QUERIES[:2], CORPUS, 10, metric="l2")
    recalls = []
    for nprobe in (1, 2, 4, 8):
        found = [check_search(ctx, q, index, 10, nprobe) for q in QUERIES[:2]]
        recalls.append(vectors.recall(found, truth))
    ctx.metric("recall_at_10_nprobe_1", round(recalls[0], 3), higher_is_better=True)
    check.that(
        all(a <= b for a, b in pairwise(recalls)),
        "recall@10 fell when nprobe grew",
        f"recall at nprobe 1, 2, 4, 8: {recalls}",
    )
    check.that(recalls[-1] == 1.0, "recall@10 at nprobe = 8 (every list) is below 1")
    check.that(recalls[0] < 1.0, "test bug: one probe already finds everything")


# Rejects drops_trailing_empty_lists: the last two centroids attract nothing.
@case("lists without vectors: in the middle and at the end of the index")
def empty_lists(ctx):
    far = np.full((2, 32), 50.0, dtype=np.float32)
    centroids = np.concatenate([CENTROIDS[:3], far[:1], CENTROIDS[3:], far[1:] + 1.0])
    index = check_build(ctx, CORPUS, centroids)
    check.that(index[1][4] == index[1][3] and index[1][-1] == index[1][-2], "test bug")
    index = oracle_index(ctx, CORPUS, centroids)
    check_search(ctx, QUERIES[0], index, 10, 10)
    # A query next to an empty list's centroid: the nearest list holds nothing.
    q = far[0] + 0.1
    ids = check_search(ctx, q, index, 10, 1)
    check.equal(len(ids), 0, label="search next to an empty list, nprobe=1: number of results")
    check_search(ctx, q, index, 10, 10)


# Rejects pads_short_results.
@case("k larger than the probed lists hold; nprobe larger than the number of lists")
def short_results(ctx):
    index = oracle_index(ctx, CORPUS, CENTROIDS)
    check_search(ctx, QUERIES[0], index, 5000, 1)
    check_search(ctx, QUERIES[0], index, 5000, 8)
    check_search(ctx, QUERIES[1], index, 10, 100)
    check_search(ctx, QUERIES[1], index, 1, 2)
    check_search(ctx, QUERIES[1], index, 300, 2)


@case("exact ties: a vector between two centroids, duplicate vectors, duplicate centroids")
def exact_ties(ctx):
    centroids = np.array([[0, 0], [4, 0], [0, 4], [4, 0]], dtype=np.float32)
    corpus = np.array(
        [[2, 0], [4, 1], [0, 2], [4, 1], [2, 2], [1, 0], [4, 1], [2, 0]], dtype=np.float32
    )
    check_build(ctx, corpus, centroids)
    index = oracle_index(ctx, corpus, centroids)
    for q in ([2.5, 0.0], [4.0, 1.0], [2.0, 2.0]):
        for nprobe in (1, 2, 4):
            check_search(ctx, np.array(q, dtype=np.float32), index, 3, nprobe)


@case("the solution's own index, searched by the solution: build and search together")
def end_to_end(ctx):
    index = check_build(ctx, CORPUS, CENTROIDS)
    for q in QUERIES[:2]:
        check_search(ctx, q, index, 10, 2)
    one = check_build(ctx, CORPUS, CENTROIDS[:1], label="build_ivf with one centroid")
    check_search(ctx, QUERIES[0], one, 10, 1)


@case("another corpus: 16 lists of 16 dimensions")
def another_corpus(ctx):
    rng = ctx.rng
    corpus, centroids = make_corpus(rng, 1200, 16, 16, spread=1.2)
    check_build(ctx, corpus, centroids)
    index = oracle_index(ctx, corpus, centroids)
    queries = vectors.queries_near(rng, corpus, [5, 900], noise=0.3, normalize=False)
    truth, _ = vectors.exact_topk(queries, corpus, 5, metric="l2")
    check_search(ctx, queries[0], index, 5, 3)
    for q, want in zip(queries, truth, strict=True):
        ids = check_search(ctx, q, index, 5, 16)
        check.equal(ids, want, label="search(k=5, nprobe=16): ids against an exact scan")
