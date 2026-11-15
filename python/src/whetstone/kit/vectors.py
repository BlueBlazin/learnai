"""vectors: seeded vector sets with known neighbors, and a distance function that counts.

Vector search problems need three things again and again: a corpus with structure
(clusters, so that an index has something to exploit), queries whose true neighbors
are known, and an exact answer to compare with. And when the problem is about doing
*less* work than a full scan, the work has to be counted.

    from whetstone.kit import vectors

    data = vectors.clustered(ctx.rng, n=2000, dim=32, clusters=16)
    data.vectors, data.centroids, data.labels        # (n, dim), (clusters, dim), (n,)
    queries = vectors.queries_near(ctx.rng, data.vectors, [3, 17, 256], noise=0.05)
    truth, scores = vectors.exact_topk(queries, data.vectors, k=10)
    vectors.recall(found, truth)                     # fraction of true neighbors found

    distance = vectors.CountingDistance(ctx.fw, metric="l2")
    d = distance(query, candidates)                  # (n_candidates,) squared L2 distances
    distance.counters.distances                      # += n_candidates

Everything is float32 and comes from the generator passed in.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from whetstone.kit.counters import Counters

METRICS = ("cosine", "dot", "l2")


@dataclass(frozen=True)
class VectorSet:
    """A corpus drawn around centroids.

    ``vectors`` is ``(n, dim)``; ``centroids`` is ``(clusters, dim)``; ``labels[i]`` is
    the centroid vector ``i`` was drawn around (not necessarily its nearest one, when
    clusters overlap).
    """

    vectors: np.ndarray
    centroids: np.ndarray
    labels: np.ndarray


def _unit(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def clustered(
    rng: np.random.Generator,
    n: int,
    dim: int,
    *,
    clusters: int = 8,
    spread: float = 0.25,
    normalize: bool = True,
) -> VectorSet:
    """``n`` vectors in ``clusters`` groups: each is a unit-length centroid plus
    Gaussian noise of standard deviation ``spread / sqrt(dim)`` per coordinate (so
    ``spread`` is roughly the noise's length relative to the centroid's). With
    ``normalize`` (the default) vectors and centroids have unit length, and cosine
    similarity, dot product, and L2 distance all rank alike."""
    if n < 1 or dim < 1 or clusters < 1 or spread < 0:
        raise ValueError("need n >= 1, dim >= 1, clusters >= 1, spread >= 0")
    centroids = _unit(rng.standard_normal((clusters, dim)))
    labels = rng.integers(0, clusters, size=n)
    points = centroids[labels] + rng.standard_normal((n, dim)) * (spread / np.sqrt(dim))
    if normalize:
        points = _unit(points)
    return VectorSet(
        points.astype(np.float32), centroids.astype(np.float32), labels.astype(np.int64)
    )


def queries_near(
    rng: np.random.Generator,
    vectors: np.ndarray,
    indices: Any,
    *,
    noise: float = 0.05,
    normalize: bool = True,
) -> np.ndarray:
    """One query per entry of ``indices``: that vector, moved by Gaussian noise of
    length about ``noise`` times its own. For small ``noise`` the vector it was made
    from is its nearest neighbor; confirm with ``exact_topk`` when a test relies on it."""
    base = np.asarray(vectors, dtype=np.float64)[np.asarray(indices, dtype=np.int64)]
    scale = noise * np.linalg.norm(base, axis=-1, keepdims=True) / np.sqrt(base.shape[-1])
    out = base + rng.standard_normal(base.shape) * scale
    return (_unit(out) if normalize else out).astype(np.float32)


def scores(queries: Any, corpus: Any, metric: str = "cosine") -> np.ndarray:
    """Similarity of every query to every corpus vector, float64 ``(Q, N)``, larger is
    closer: cosine similarity, dot product, or the negated squared L2 distance."""
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {', '.join(METRICS)}, got {metric!r}")
    q = np.atleast_2d(np.asarray(queries, dtype=np.float64))
    c = np.asarray(corpus, dtype=np.float64)
    if c.ndim != 2 or q.shape[-1] != c.shape[-1]:
        raise ValueError(f"queries {q.shape} and corpus {c.shape} do not fit together")
    if metric == "cosine":
        return _unit(q) @ _unit(c).T
    if metric == "dot":
        return q @ c.T
    return -((q * q).sum(-1)[:, None] - 2.0 * (q @ c.T) + (c * c).sum(-1)[None, :])


def exact_topk(
    queries: Any, corpus: Any, k: int, *, metric: str = "cosine"
) -> tuple[np.ndarray, np.ndarray]:
    """The true ``k`` nearest corpus vectors of every query: ``(indices, scores)``.

    Both are ``(Q, min(k, N))``, best first; equal scores are ordered by the lower
    index. ``indices`` is int64; ``scores`` is float32 (see ``scores`` for the three
    metrics). Computed in float64 by a full scan: the oracle, not a baseline to race.
    """
    if k < 1:
        raise ValueError("k must be at least 1")
    table = scores(queries, corpus, metric)
    order = np.argsort(-table, axis=-1, kind="stable")[:, :k]
    return order.astype(np.int64), np.take_along_axis(table, order, axis=-1).astype(np.float32)


def recall(found: Any, truth: Any) -> float:
    """The fraction of the true neighbors that were found, averaged over queries.

    ``found`` and ``truth`` are ``(Q, k)`` index arrays (or lists of rows of any
    lengths); order within a row does not matter.
    """
    rows = list(zip(found, truth, strict=True))
    if not rows:
        raise ValueError("recall of no queries")
    hits = [len(set(map(int, f)) & set(map(int, t))) / len(t) for f, t in rows]
    return float(np.mean(hits))


class CountingDistance:
    """A distance function that counts how many vectors it was asked about.

        distance = CountingDistance(ctx.fw, metric="l2")
        d = distance(query, candidates)          # (N,) for a query (dim,), candidates (N, dim)
        d = distance(queries, candidates)        # (Q, N) for queries (Q, dim)

    ``metric`` is ``"l2"`` (squared Euclidean distance: smaller is closer), ``"dot"``,
    or ``"cosine"`` (similarities: larger is closer). The result is a float32 tensor
    of the framework. Counters: ``calls``, and ``distances``, the number of
    (query, candidate) pairs computed, ``Q * N`` per call. An index that scans only
    the lists it probes shows it here; ``N`` may be 0.
    """

    def __init__(self, fw: Any, metric: str = "l2") -> None:
        if metric not in METRICS:
            raise ValueError(f"metric must be one of {', '.join(METRICS)}, got {metric!r}")
        self.fw = fw
        self.metric = metric
        self.counters = Counters("calls", "distances")

    def with_framework(self, fw: Any) -> CountingDistance:
        """The same function for another framework, with fresh counters."""
        return CountingDistance(fw, self.metric)

    def __call__(self, queries: Any, candidates: Any) -> Any:
        q = np.asarray(self.fw.numpy(queries) if self.fw.is_tensor(queries) else queries)
        c = np.asarray(self.fw.numpy(candidates) if self.fw.is_tensor(candidates) else candidates)
        if q.ndim not in (1, 2) or c.ndim != 2 or q.shape[-1] != c.shape[-1]:
            raise ValueError(
                f"expected queries (dim,) or (Q, dim) and candidates (N, dim); got "
                f"{q.shape} and {c.shape}"
            )
        single = q.ndim == 1
        table = np.zeros((1 if single else q.shape[0], c.shape[0]), dtype=np.float32)
        if c.shape[0]:
            table = scores(q, c, self.metric)
            table = (-table if self.metric == "l2" else table).astype(np.float32)
        self.counters.add("calls")
        self.counters.add("distances", int(table.size))
        return self.fw.tensor(table[0] if single else table)

    def __repr__(self) -> str:
        return f"CountingDistance({self.metric!r}, {self.counters!r})"
