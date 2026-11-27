# BUG: masked_max takes the max over x * mask: a dropped step counts as 0, so a row whose kept values are all negative returns 0, a row that keeps nothing returns 0 instead of -inf, and a dropped inf or NaN becomes NaN.
import torch


def fill_masked(scores: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    return torch.where(keep, scores, float("-inf"))


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    kept = torch.where(mask[:, :, None], x, 0.0)
    count = mask.sum(dim=1)[:, None]  # (B, 1): kept steps per row
    return kept.sum(dim=1) / count.clamp(min=1)


def masked_max(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (x * mask[:, :, None]).amax(dim=1)
