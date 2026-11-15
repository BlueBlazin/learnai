import mlx.core as mx


def build_ivf(
    corpus: mx.array, centroids: mx.array
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """Group the corpus by nearest centroid.

    corpus: float32, shape (N, D). centroids: float32, shape (C, D).
    Every vector goes to the list of its nearest centroid (squared L2 distance; ties
    to the lower centroid id). Returns the tuple (centroids, offsets, ids, vectors):
      offsets: integer, shape (C + 1,). List j is positions offsets[j] to offsets[j + 1] - 1.
      ids: integer, shape (N,). Corpus indices grouped by list, ascending within a list.
      vectors: float32, shape (N, D). corpus[ids].
    """
    raise NotImplementedError


def search(
    q: mx.array, index: tuple[mx.array, mx.array, mx.array, mx.array], k: int, nprobe: int, distance
) -> tuple[mx.array, mx.array]:
    """The k nearest vectors among the nprobe lists whose centroids are nearest to q.

    q: float32, shape (D,). index: the tuple returned by build_ivf.
    distance: a kit CountingDistance with metric "l2"; distance(q, rows) returns the
        squared L2 distances from q to the rows, float32, shape (len(rows),).
    Every distance must come from `distance`: one call for the centroids, and the
    vectors of the probed lists only.
    Returns (ids, distances): corpus indices and their distances, nearest first;
    min(k, number of vectors in the probed lists) entries.
    """
    raise NotImplementedError
