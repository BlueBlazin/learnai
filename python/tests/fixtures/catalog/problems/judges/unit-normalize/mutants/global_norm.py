# BUG: divides by the norm of the whole tensor instead of the norm along the axis.
import torch


def normalize(x, axis=-1):
    return x / torch.sqrt((x * x).sum())
