# BUG: exclusive_cumsum shifts the inclusive sums with roll, which wraps: position 0 holds the total of the row instead of 0. through_first_eos is written without it and is not affected.
import torch


def exclusive_cumsum(x: torch.Tensor) -> torch.Tensor:
    return torch.cumsum(x, dim=-1).roll(1, dims=-1)


def first_true(mask: torch.Tensor) -> torch.Tensor:
    seen = torch.cumsum(mask, dim=1)  # (B, T) int64
    return (seen == 0).sum(dim=1)


def through_first_eos(ids: torch.Tensor, eos: int) -> torch.Tensor:
    is_eos = (ids == eos).to(torch.int64)
    return torch.cumsum(is_eos, dim=1) - is_eos == 0
