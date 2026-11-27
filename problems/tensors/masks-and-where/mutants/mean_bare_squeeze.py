# BUG: masked_mean keeps the step axis through the reduction and removes it with a bare squeeze(), which also removes a batch axis or a feature axis of size 1.
import torch


def fill_masked(scores: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    return torch.where(keep, scores, float("-inf"))


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    kept = torch.where(mask[:, :, None], x, 0.0)
    count = mask.sum(dim=1)[:, None, None]
    mean = kept.sum(dim=1, keepdim=True) / count.clamp(min=1)
    return mean.squeeze()


def masked_max(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return torch.where(mask[:, :, None], x, float("-inf")).amax(dim=1)
