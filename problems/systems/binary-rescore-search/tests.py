"""Tests for binary-rescore-search.

Codes, indices, and Hamming distances are exact and are compared with numpy
(`np.packbits`, a brute-force popcount) as well as with the oracle. The rescoring
scores come from the kit's `CountingDistance`, which also counts how many rows the
solution asked it to score.

Shapes are shared across cases for JAX's sake: the corpus is (2000, 64) with
(2000, 8) codes, or the tiny ones of the worked example and the tie cases.
"""

import numpy as np

from learnai.judge import case, check
from learnai.kit import vectors

TOL = {"rtol": 1e-4, "atol": 1e-5}  # the tolerance stated in Constraints
POPCOUNT = np.array([byte.bit_count() for byte in range(256)])


def to_int8(corpus):
    """Scalar quantization with one global scale: the largest magnitude maps to 127."""
    return np.round(corpus * (127.0 / np.abs(corpus).max())).astype(np.int8)


def make_corpus(rng, n=2000, dim=64, clusters=100, n_queries=4):
    """(float corpus, codes, int8 vectors, float queries near corpus vectors)."""
    data = vectors.clustered(rng, n=n, dim=dim, clusters=clusters, spread=0.5)
    queries = vectors.queries_near(rng, data.vectors, rng.integers(0, n, n_queries), noise=0.1)
    return data.vectors, np.packbits(data.vectors > 0, axis=-1), to_int8(data.vectors), queries


def hamming(q_bits, corpus_bits):
    return POPCOUNT[np.bitwise_xor(corpus_bits, q_bits)].sum(axis=-1)


def check_binarize(ctx, e):
    got = ctx.check("binarize", e)
    check.that(got.dtype == np.uint8, f"binarize: got dtype {got.dtype}; codes are uint8")
    check.equal(got, np.packbits(np.asarray(e) > 0, axis=-1), label="binarize")
    return got


def check_hamming(ctx, q_bits, corpus_bits, m):
    label = f"hamming_topk(m={m})"
    got = ctx.call("hamming_topk", q_bits, corpus_bits, m)
    check.that(
        isinstance(got, tuple) and len(got) == 2,
        f"{label}: must return a tuple (indices, distances)",
        f"got: {type(got).__name__}",
    )
    indices, distances = got
    table = hamming(q_bits, corpus_bits)
    order = np.argsort(table, kind="stable")[:m]
    check.equal(distances, table[order], label=f"{label}: distances")
    check.equal(indices, order, label=f"{label}: indices")
    want = ctx.expected("hamming_topk", q_bits, corpus_bits, m)
    check.that(np.array_equal(want[0], order), "test bug: the oracle disagrees with numpy")


def check_search(ctx, q, corpus_bits, corpus_int8, k, oversample, *, metric=False):
    label = f"search(k={k}, oversample={oversample})"
    distance = vectors.CountingDistance(ctx.fw, metric="dot")
    got = ctx.call("search", q, corpus_bits, corpus_int8, k, oversample, distance)
    check.that(
        isinstance(got, tuple) and len(got) == 2,
        f"{label}: must return a tuple (indices, scores)",
        f"got: {type(got).__name__}",
    )
    indices, scores = got
    oracle_distance = vectors.CountingDistance(ctx.oracle_fw, metric="dot")
    want_indices, want_scores = ctx.expected(
        "search", q, corpus_bits, corpus_int8, k, oversample, oracle_distance
    )
    check.equal(indices, want_indices, label=f"{label}: indices")
    check.allclose(scores, want_scores, **TOL, dtype=True, label=f"{label}: scores")
    limit = min(k * oversample, corpus_bits.shape[0])
    if metric:
        check.cost(distance.counters, "distances", at_most=limit, unit="vectors",
                   metric="rescored_vectors")  # fmt: skip
    else:
        scored = distance.counters.distances
        check.that(
            scored <= limit,
            f"{label}: {scored} vectors were scored at full precision, which exceeds the "
            f"limit of {limit} (k * oversample candidates)",
        )
    return indices


EXAMPLE_INT8 = np.array(
    [
        [10, 10, 10, 10, -10, -10, -10, -10],
        [100, -5, 5, 5, -5, -5, -5, -5],
        [-100, 5, 5, 5, -5, -5, -5, -5],
        [-10, -10, -10, -10, 10, 10, 10, 10],
        [50, 50, -5, -5, 5, -5, -5, -5],
    ],
    dtype=np.int8,
)
EXAMPLE_BITS = np.array([[240], [176], [112], [15], [200]], dtype=np.uint8)
EXAMPLE_Q = np.array([0.75, 0.125, 0.125, 0.125, -0.125, -0.125, -0.125, -0.125], np.float32)


@case("the worked example: one byte of code, five documents", public=True)
def worked_example(ctx):
    e = np.array([0.5, -1.0, 0.0, 2.0, -0.1, 3.0, 0.0, -4.0], dtype=np.float32)
    check.equal(check_binarize(ctx, e), np.array([148]), label="binarize(e)")
    check.equal(check_binarize(ctx, EXAMPLE_Q), np.array([240]), label="binarize(q)")
    q_bits = np.array([240], dtype=np.uint8)
    indices, distances = ctx.call("hamming_topk", q_bits, EXAMPLE_BITS, 3)
    check.equal(indices, np.array([0, 1, 2]), label="hamming_topk(m=3): indices")
    check.equal(distances, np.array([0, 1, 1]), label="hamming_topk(m=3): distances")
    for (k, oversample), (want_indices, want_scores) in {
        (1, 1): ([0], [16.25]),
        (1, 2): ([1], [78.125]),
        (2, 2): ([1, 4], [78.125, 43.75]),
    }.items():
        label = f"search(k={k}, oversample={oversample})"
        distance = vectors.CountingDistance(ctx.fw, metric="dot")
        indices, scores = ctx.call(
            "search", EXAMPLE_Q, EXAMPLE_BITS, EXAMPLE_INT8, k, oversample, distance
        )
        check.equal(indices, np.array(want_indices), label=f"{label}: indices")
        check.allclose(scores, want_scores, **TOL, label=f"{label}: scores")
        check.cost(distance.counters, "distances", at_most=k * oversample, unit="vectors",
                   metric="rescored_vectors")  # fmt: skip


# Rejects zero_is_positive and lsb_first.
@case("binarize: matches numpy.packbits, with zeros, and for a matrix and a vector",
      public=True)  # fmt: skip
def binarize_packbits(ctx):
    e = ctx.randn(2000, 64)
    e[::7, ::5] = 0.0
    e[3, :8] = -0.0
    check_binarize(ctx, e)
    check_binarize(ctx, e[0])
    check_binarize(ctx, np.zeros(64, dtype=np.float32))
    check_binarize(ctx, np.ones(64, dtype=np.float32))


# Rejects hamming_counts_bytes.
@case("hamming_topk: the nearest codes of a 2,000-document corpus", public=True)
def hamming_nearest(ctx):
    _, bits, _, queries = make_corpus(ctx.rng)
    for q in queries[:2]:
        q_bits = np.packbits(q > 0)
        check_hamming(ctx, q_bits, bits, 40)
    check_hamming(ctx, bits[17], bits, 10)


# Rejects rescores_everything (cost), ascending_similarity, no_reordering, and
# ignores_oversample.
@case("search: 40 candidates rescored for the top 10; only the candidates are scored",
      public=True)  # fmt: skip
def search_rescored(ctx):
    _, bits, int8, queries = make_corpus(ctx.rng)
    for i, q in enumerate(queries):
        check_search(ctx, q, bits, int8, 10, 4, metric=i == 0)


@case("recall@10 against exact float search is at least 0.9 with oversample 4", public=True)
def recall_level(ctx):
    corpus, bits, int8, queries = make_corpus(ctx.rng, n_queries=20)
    truth, _ = vectors.exact_topk(queries, corpus, 10, metric="dot")
    found = [check_search(ctx, q, bits, int8, 10, 4) for q in queries]
    recall = vectors.recall(found, truth)
    ctx.metric("recall_at_10", round(recall, 3), higher_is_better=True)
    check.that(
        recall >= 0.9,
        f"recall@10 is {recall:.3f}, which is below the level the statement gives",
        "expected at least 0.9 on this corpus (the reference reaches 0.96)",
    )
    hamming_only = [np.argsort(hamming(np.packbits(q > 0), bits), kind="stable")[:10]
                    for q in queries]  # fmt: skip
    coarse = vectors.recall(hamming_only, truth)
    ctx.metric("recall_at_10_hamming_only", round(coarse, 3), higher_is_better=True)
    check.that(coarse < recall, "test bug: rescoring did not help")


# Rejects hamming_ties_arbitrary: with 8-bit codes there are only nine distances.
@case("hamming_topk ties: one-byte codes, equal distances ordered by index")
def hamming_ties(ctx):
    rng = ctx.rng
    bits = rng.integers(0, 256, size=(200, 1)).astype(np.uint8)
    bits[150:160] = bits[20:30]
    for m in (10, 40):
        check_hamming(ctx, bits[25], bits, m)
        check_hamming(ctx, np.array([0], dtype=np.uint8), bits, m)


@case("m and k * oversample larger than the corpus; a corpus of one document")
def more_than_corpus(ctx):
    rng = ctx.rng
    bits = rng.integers(0, 256, size=(200, 1)).astype(np.uint8)
    check_hamming(ctx, bits[3], bits, 200)
    check_hamming(ctx, bits[3], bits, 1000)
    int8 = rng.integers(-127, 128, size=(5, 8)).astype(np.int8)
    codes = np.packbits(int8 > 0, axis=-1)
    q = ctx.randn(8)
    check_search(ctx, q, codes, int8, 2, 4)
    check_search(ctx, q, codes, int8, 10, 4)
    check_search(ctx, q, codes[:1], int8[:1], 3, 2)


# Duplicated documents have equal codes and equal scores: candidate order decides.
@case("search ties: duplicated documents keep the candidate order")
def search_ties(ctx):
    rng = ctx.rng
    int8 = rng.integers(-127, 128, size=(5, 8)).astype(np.int8)
    int8 = int8[[0, 1, 2, 0, 1, 3, 0, 4, 2, 2]]
    codes = np.packbits(int8 > 0, axis=-1)
    for row in (0, 1, 2):
        q = int8[row].astype(np.float32) / 64.0
        check_search(ctx, q, codes, int8, 4, 2)
        check_search(ctx, q, codes, int8, 10, 1)


@case("oversample 1, k = 1, and several oversampling factors on one query")
def oversampling_factors(ctx):
    _, bits, int8, queries = make_corpus(ctx.rng)
    q = queries[0]
    for k, oversample in ((10, 1), (1, 1), (1, 40), (10, 2), (5, 8)):
        check_search(ctx, q, bits, int8, k, oversample)


@case("random corpora and queries far from every document")
def random_sweep(ctx):
    rng = ctx.rng
    for _ in range(3):
        _, bits, int8, _ = make_corpus(rng, clusters=int(rng.integers(4, 200)))
        q = ctx.randn(64)
        check_hamming(ctx, np.packbits(q > 0), bits, 40)
        check_search(ctx, q, bits, int8, 10, 4)
