# BUG: log_softmax is computed as log(softmax(x)); probabilities that underflow to 0 become -inf although their logarithm is representable.
import torch


def softmax(x: torch.Tensor, axis: int = -1) -> torch.Tensor:
    shifted = x - torch.amax(x, dim=axis, keepdim=True)
    e = torch.exp(shifted)
    return e / e.sum(dim=axis, keepdim=True)


def log_softmax(x: torch.Tensor, axis: int = -1) -> torch.Tensor:
    return torch.log(softmax(x, axis))
