import mlx.core as mx


def build_ivf(
    corpus: mx.array, centroids: mx.array
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    # Squared distance of every vector to every centroid, (N, C). A stable sort of
    # each row puts the lower centroid id first among equal distances.
    gaps = corpus[:, None, :] - centroids[None, :, :]
    assigned = mx.argsort((gaps * gaps).sum(axis=-1), axis=1)[:, 0]

    # A stable sort by list number groups the vectors and keeps ascending ids inside
    # each list. Storing the vectors in that order makes every list one contiguous
    # block, which is what a scan wants.
    ids = mx.argsort(assigned)  # stable
    # One count per list, also for a list without vectors: its range is empty.
    lists = mx.arange(centroids.shape[0])
    sizes = (assigned[:, None] == lists[None, :]).sum(axis=0)
    offsets = mx.concatenate([mx.zeros(1, dtype=sizes.dtype), mx.cumsum(sizes)])
    return centroids, offsets, ids, corpus[ids]


def search(
    q: mx.array,
    index: tuple[mx.array, mx.array, mx.array, mx.array],
    k: int,
    nprobe: int,
    distance,
) -> tuple[mx.array, mx.array]:
    centroids, offsets, ids, vectors = index
    # Coarse step: C distances decide which lists are worth scanning.
    to_centroids = distance(q, centroids)
    probe = mx.argsort(to_centroids)[:nprobe].tolist()  # stable; nearest first

    # The positions of the probed lists, in scan order. An empty list adds nothing.
    bounds = offsets.tolist()
    positions = mx.concatenate([mx.arange(bounds[j], bounds[j + 1]) for j in probe])

    # Fine step: distances to the vectors of those lists, and to no others.
    distances = distance(q, vectors[positions])
    best = mx.argsort(distances)[:k]  # fewer than k if the lists are short
    return ids[positions[best]], distances[best]
