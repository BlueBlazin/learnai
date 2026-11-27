# BUG: exclusive_cumsum accumulates along axis 0 instead of the last axis; it is right only for an input with one axis. through_first_eos is written without it and is not affected.
import torch


def exclusive_cumsum(x: torch.Tensor) -> torch.Tensor:
    total = torch.cumsum(x, dim=0)
    zero = torch.zeros_like(x[:1])
    return torch.cat([zero, total[:-1]], dim=0)


def first_true(mask: torch.Tensor) -> torch.Tensor:
    seen = torch.cumsum(mask, dim=1)  # (B, T) int64
    return (seen == 0).sum(dim=1)


def through_first_eos(ids: torch.Tensor, eos: int) -> torch.Tensor:
    is_eos = (ids == eos).to(torch.int64)
    return torch.cumsum(is_eos, dim=1) - is_eos == 0
