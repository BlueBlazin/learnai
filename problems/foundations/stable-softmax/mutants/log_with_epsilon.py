# BUG: log_softmax adds an epsilon inside the log to hide -inf; every log-probability below about -18.4 is clamped to the wrong value.
import torch


def softmax(x: torch.Tensor, axis: int = -1) -> torch.Tensor:
    shifted = x - torch.amax(x, dim=axis, keepdim=True)
    e = torch.exp(shifted)
    return e / e.sum(dim=axis, keepdim=True)


def log_softmax(x: torch.Tensor, axis: int = -1) -> torch.Tensor:
    return torch.log(softmax(x, axis) + 1e-8)
