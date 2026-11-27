# BUG: masked_max replaces the -inf of a row that keeps no step with 0.
import torch


def fill_masked(scores: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    return torch.where(keep, scores, float("-inf"))


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    kept = torch.where(mask[:, :, None], x, 0.0)
    count = mask.sum(dim=1)[:, None]  # (B, 1): kept steps per row
    return kept.sum(dim=1) / count.clamp(min=1)


def masked_max(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    best = torch.where(mask[:, :, None], x, float("-inf")).amax(dim=1)
    return torch.where(mask.any(dim=1)[:, None], best, 0.0)
