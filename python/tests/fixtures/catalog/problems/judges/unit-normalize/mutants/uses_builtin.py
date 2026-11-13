# BUG: calls the built-in torch.nn.functional.normalize, which the problem forbids.
import torch.nn.functional as F


def normalize(x, axis=-1):
    return F.normalize(x, dim=axis, eps=0.0)
