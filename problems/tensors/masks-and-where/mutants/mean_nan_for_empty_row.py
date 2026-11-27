# BUG: masked_mean divides by the count without guarding a count of 0: a row that keeps no step returns 0 / 0 = NaN instead of zeros.
import torch


def fill_masked(scores: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    return torch.where(keep, scores, float("-inf"))


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    kept = torch.where(mask[:, :, None], x, 0.0)
    count = mask.sum(dim=1)[:, None]
    return kept.sum(dim=1) / count


def masked_max(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return torch.where(mask[:, :, None], x, float("-inf")).amax(dim=1)
