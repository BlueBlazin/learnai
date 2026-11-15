# BUG: vectors are assigned to the centroid with the largest inner product instead of the smallest L2 distance; for vectors that are not unit length that is a different centroid.
import torch


def build_ivf(
    corpus: torch.Tensor, centroids: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    assigned = (corpus @ centroids.T).argmax(dim=1)

    # A stable sort by list number groups the vectors and keeps ascending ids inside
    # each list. Storing the vectors in that order makes every list one contiguous
    # block, which is what a scan wants.
    ids = torch.argsort(assigned, stable=True)
    # minlength: a list without vectors still needs its (empty) range.
    sizes = torch.bincount(assigned, minlength=centroids.shape[0])
    offsets = torch.cat([sizes.new_zeros(1), sizes.cumsum(dim=0)])
    return centroids, offsets, ids, corpus[ids]


def search(
    q: torch.Tensor,
    index: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    k: int,
    nprobe: int,
    distance,
) -> tuple[torch.Tensor, torch.Tensor]:
    centroids, offsets, ids, vectors = index
    # Coarse step: C distances decide which lists are worth scanning.
    to_centroids = distance(q, centroids)
    probe = torch.argsort(to_centroids, stable=True)[:nprobe].tolist()  # nearest first

    # The positions of the probed lists, in scan order. An empty list adds nothing.
    bounds = offsets.tolist()
    positions = torch.cat([torch.arange(bounds[j], bounds[j + 1]) for j in probe])

    # Fine step: distances to the vectors of those lists, and to no others.
    distances = distance(q, vectors[positions])
    best = torch.argsort(distances, stable=True)[:k]  # fewer than k if the lists are short
    return ids[positions[best]], distances[best]
