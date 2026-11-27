# BUG: rank_descending sorts along axis 0: it orders the rows within each column instead of the columns within each row.
import torch


def rank_descending(x: torch.Tensor) -> torch.Tensor:
    return torch.argsort(x, dim=0, descending=True, stable=True)


def inverse_permutation(perm: torch.Tensor) -> torch.Tensor:
    ranks = torch.arange(perm.shape[1]).expand_as(perm)  # ranks[b, k] = k
    return torch.zeros_like(perm).scatter(1, perm, ranks)


def keep_to_original_order(keep_sorted: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(keep_sorted).scatter(1, order, keep_sorted)
