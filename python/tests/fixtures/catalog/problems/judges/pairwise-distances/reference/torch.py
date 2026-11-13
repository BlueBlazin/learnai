def pairwise_sq_dists(a, b):
    diff = a[:, None, :] - b[None, :, :]
    return (diff * diff).sum(dim=-1)
