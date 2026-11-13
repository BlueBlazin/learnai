# BUG: ignores the axis argument and always normalizes over the last dimension.
import torch


def softmax(x: torch.Tensor, axis: int = -1) -> torch.Tensor:
    shifted = x - torch.amax(x, dim=-1, keepdim=True)
    e = torch.exp(shifted)
    return e / e.sum(dim=-1, keepdim=True)


def log_softmax(x: torch.Tensor, axis: int = -1) -> torch.Tensor:
    shifted = x - torch.amax(x, dim=-1, keepdim=True)
    return shifted - torch.log(torch.exp(shifted).sum(dim=-1, keepdim=True))
