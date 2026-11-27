# BUG: first_true keeps the position axis through the sum and removes it with a bare squeeze(), which also removes the batch axis of a single row.
import torch


def exclusive_cumsum(x: torch.Tensor) -> torch.Tensor:
    total = torch.cumsum(x, dim=-1)
    zero = torch.zeros_like(x[..., :1])  # a slice, so the axis is kept with size 1
    return torch.cat([zero, total[..., :-1]], dim=-1)


def first_true(mask: torch.Tensor) -> torch.Tensor:
    seen = torch.cumsum(mask, dim=1)
    return (seen == 0).sum(dim=1, keepdim=True).squeeze()


def through_first_eos(ids: torch.Tensor, eos: int) -> torch.Tensor:
    is_eos = (ids == eos).to(torch.int64)
    return exclusive_cumsum(is_eos) == 0
