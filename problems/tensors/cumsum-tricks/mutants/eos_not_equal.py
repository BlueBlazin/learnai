# BUG: through_first_eos returns ids != eos: it is false only at the eos tokens themselves, so the first eos is dropped and every other token after it is true again.
import torch


def exclusive_cumsum(x: torch.Tensor) -> torch.Tensor:
    total = torch.cumsum(x, dim=-1)
    zero = torch.zeros_like(x[..., :1])  # a slice, so the axis is kept with size 1
    return torch.cat([zero, total[..., :-1]], dim=-1)


def first_true(mask: torch.Tensor) -> torch.Tensor:
    seen = torch.cumsum(mask, dim=1)  # (B, T) int64
    return (seen == 0).sum(dim=1)


def through_first_eos(ids: torch.Tensor, eos: int) -> torch.Tensor:
    return ids != eos
