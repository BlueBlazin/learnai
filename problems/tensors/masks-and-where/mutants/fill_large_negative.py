# BUG: fill_masked fills with -1e9 instead of -inf.
import torch


def fill_masked(scores: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    return torch.where(keep, scores, -1e9)


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    kept = torch.where(mask[:, :, None], x, 0.0)
    count = mask.sum(dim=1)[:, None]  # (B, 1): kept steps per row
    return kept.sum(dim=1) / count.clamp(min=1)


def masked_max(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return torch.where(mask[:, :, None], x, float("-inf")).amax(dim=1)
