# BUG: first_true counts every false entry of the row instead of the false entries before the first true one; it is right only when nothing false follows the first true entry.
import torch


def exclusive_cumsum(x: torch.Tensor) -> torch.Tensor:
    total = torch.cumsum(x, dim=-1)
    zero = torch.zeros_like(x[..., :1])  # a slice, so the axis is kept with size 1
    return torch.cat([zero, total[..., :-1]], dim=-1)


def first_true(mask: torch.Tensor) -> torch.Tensor:
    return (~mask).sum(dim=1)


def through_first_eos(ids: torch.Tensor, eos: int) -> torch.Tensor:
    is_eos = (ids == eos).to(torch.int64)
    return exclusive_cumsum(is_eos) == 0
