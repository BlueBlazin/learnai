import jax
import jax.numpy as jnp


def build_ivf(
    corpus: jax.Array, centroids: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    # Squared distance of every vector to every centroid, (N, C). argmin returns the
    # first minimum, so a tie goes to the lower centroid id.
    gaps = corpus[:, None, :] - centroids[None, :, :]
    assigned = (gaps * gaps).sum(axis=-1).argmin(axis=1)

    # A stable sort by list number groups the vectors and keeps ascending ids inside
    # each list. Storing the vectors in that order makes every list one contiguous
    # block, which is what a scan wants.
    ids = jnp.argsort(assigned)  # stable
    # minlength: a list without vectors still needs its (empty) range.
    sizes = jnp.bincount(assigned, minlength=centroids.shape[0])
    offsets = jnp.concatenate([jnp.zeros(1, dtype=sizes.dtype), jnp.cumsum(sizes)])
    return centroids, offsets, ids, corpus[ids]


def search(
    q: jax.Array,
    index: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    k: int,
    nprobe: int,
    distance,
) -> tuple[jax.Array, jax.Array]:
    centroids, offsets, ids, vectors = index
    # Coarse step: C distances decide which lists are worth scanning.
    to_centroids = distance(q, centroids)
    probe = jnp.argsort(to_centroids)[:nprobe].tolist()  # stable; nearest first

    # The positions of the probed lists, in scan order. An empty list adds nothing.
    bounds = offsets.tolist()
    positions = jnp.concatenate([jnp.arange(bounds[j], bounds[j + 1]) for j in probe])

    # Fine step: distances to the vectors of those lists, and to no others.
    distances = distance(q, vectors[positions])
    best = jnp.argsort(distances)[:k]  # fewer than k if the lists are short
    return ids[positions[best]], distances[best]
