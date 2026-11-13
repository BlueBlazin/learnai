# BUG: reduces without keepdim; the max and the sum then broadcast along the wrong axis (silently, for square inputs) or fail to broadcast at all.
import torch


def softmax(x: torch.Tensor, axis: int = -1) -> torch.Tensor:
    shifted = x - torch.amax(x, dim=axis)
    e = torch.exp(shifted)
    return e / e.sum(dim=axis)


def log_softmax(x: torch.Tensor, axis: int = -1) -> torch.Tensor:
    shifted = x - torch.amax(x, dim=axis)
    return shifted - torch.log(torch.exp(shifted).sum(dim=axis))
