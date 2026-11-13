import torch


def softmax(x: torch.Tensor, axis: int = -1) -> torch.Tensor:
    # Shift so the largest logit along the axis is 0: exponents are <= 0 (no overflow)
    # and the largest term is exp(0) = 1 (the denominator is never 0).
    shifted = x - torch.amax(x, dim=axis, keepdim=True)
    e = torch.exp(shifted)
    return e / e.sum(dim=axis, keepdim=True)


def log_softmax(x: torch.Tensor, axis: int = -1) -> torch.Tensor:
    # Stay in log space: the sum is in [1, n], so its log is finite and small.
    shifted = x - torch.amax(x, dim=axis, keepdim=True)
    return shifted - torch.log(torch.exp(shifted).sum(dim=axis, keepdim=True))
