import mlx.core as mx


def pairwise_sq_dists(a, b):
    diff = a[:, None, :] - b[None, :, :]
    return mx.sum(diff * diff, axis=-1)
