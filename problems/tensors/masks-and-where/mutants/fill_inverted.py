# BUG: fill_masked fills the kept entries and returns the dropped ones: the mask is used without negation as the place to fill.
import torch


def fill_masked(scores: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    return scores.masked_fill(keep, float("-inf"))


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    kept = torch.where(mask[:, :, None], x, 0.0)
    count = mask.sum(dim=1)[:, None]  # (B, 1): kept steps per row
    return kept.sum(dim=1) / count.clamp(min=1)


def masked_max(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return torch.where(mask[:, :, None], x, float("-inf")).amax(dim=1)
