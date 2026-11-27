# BUG: through_first_eos tests that the inclusive count of eos tokens is at most 1, which stays true after the first eos until the second one.
import torch


def exclusive_cumsum(x: torch.Tensor) -> torch.Tensor:
    total = torch.cumsum(x, dim=-1)
    zero = torch.zeros_like(x[..., :1])  # a slice, so the axis is kept with size 1
    return torch.cat([zero, total[..., :-1]], dim=-1)


def first_true(mask: torch.Tensor) -> torch.Tensor:
    seen = torch.cumsum(mask, dim=1)  # (B, T) int64
    return (seen == 0).sum(dim=1)


def through_first_eos(ids: torch.Tensor, eos: int) -> torch.Tensor:
    is_eos = (ids == eos).to(torch.int64)
    return torch.cumsum(is_eos, dim=1) <= 1
