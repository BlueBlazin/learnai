# BUG: masked_mean zeroes the dropped steps by multiplying x by the mask instead of selecting with where: a dropped inf, -inf, or NaN becomes NaN (0 * inf), and the mean of that row is NaN.
import torch


def fill_masked(scores: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    return torch.where(keep, scores, float("-inf"))


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    kept = x * mask[:, :, None]
    count = mask.sum(dim=1)[:, None]
    return kept.sum(dim=1) / count.clamp(min=1)


def masked_max(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return torch.where(mask[:, :, None], x, float("-inf")).amax(dim=1)
