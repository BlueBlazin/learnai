# BUG: correct values, but computed with Python loops over every pair of rows.
import torch


def pairwise_sq_dists(a, b):
    out = torch.empty(a.shape[0], b.shape[0])
    for i in range(a.shape[0]):
        for j in range(b.shape[0]):
            d = a[i] - b[j]
            out[i, j] = (d * d).sum()
    return out
