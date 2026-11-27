# BUG: inverse_permutation takes the argsort along axis 0 instead of along each row.
import torch


def rank_descending(x: torch.Tensor) -> torch.Tensor:
    return torch.argsort(x, dim=-1, descending=True, stable=True)


def inverse_permutation(perm: torch.Tensor) -> torch.Tensor:
    return torch.argsort(perm, dim=0)


def keep_to_original_order(keep_sorted: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(keep_sorted).scatter(1, order, keep_sorted)
