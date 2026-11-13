# BUG: subtracts the global maximum instead of the per-slice maximum; slices far below the global maximum still underflow to 0/0.
import torch


def softmax(x: torch.Tensor, axis: int = -1) -> torch.Tensor:
    shifted = x - x.max()
    e = torch.exp(shifted)
    return e / e.sum(dim=axis, keepdim=True)


def log_softmax(x: torch.Tensor, axis: int = -1) -> torch.Tensor:
    shifted = x - x.max()
    return shifted - torch.log(torch.exp(shifted).sum(dim=axis, keepdim=True))
