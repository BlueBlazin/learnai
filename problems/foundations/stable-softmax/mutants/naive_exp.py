# BUG: exponentiates the raw logits (textbook formula); overflows to inf/inf = NaN for large logits and underflows to 0/0 for very negative ones.
import torch


def softmax(x: torch.Tensor, axis: int = -1) -> torch.Tensor:
    e = torch.exp(x)
    return e / e.sum(dim=axis, keepdim=True)


def log_softmax(x: torch.Tensor, axis: int = -1) -> torch.Tensor:
    return x - torch.log(torch.exp(x).sum(dim=axis, keepdim=True))
