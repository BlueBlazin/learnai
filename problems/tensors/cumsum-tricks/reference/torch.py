import torch


def exclusive_cumsum(x: torch.Tensor) -> torch.Tensor:
    # The inclusive sums, moved one place to the right: position i gets the sum that
    # ended at i - 1, position 0 gets 0, and the last inclusive sum is dropped.
    total = torch.cumsum(x, dim=-1)
    zero = torch.zeros_like(x[..., :1])  # a slice, so the axis is kept with size 1
    return torch.cat([zero, total[..., :-1]], dim=-1)


def first_true(mask: torch.Tensor) -> torch.Tensor:
    # The running count of true entries is 0 exactly at the positions before the first
    # one. There are as many of those as the index of the first true entry, and T of
    # them when the row has none.
    seen = torch.cumsum(mask, dim=1)  # (B, T) int64
    return (seen == 0).sum(dim=1)


def through_first_eos(ids: torch.Tensor, eos: int) -> torch.Tensor:
    # Position t is kept when no eos comes strictly before it. The exclusive count
    # leaves the first eos itself at 0, so it is kept, and everything after it is not.
    is_eos = (ids == eos).to(torch.int64)
    return exclusive_cumsum(is_eos) == 0
