import torch


def normalize(x, axis=-1):
    return x / torch.sqrt((x * x).sum(dim=axis, keepdim=True))
