# BUG: rank_descending returns the rank of every column (the argsort of the argsort) instead of the column at every rank; the two agree only when a row's order is its own inverse.
import torch


def rank_descending(x: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(x, dim=-1, descending=True, stable=True)
    return torch.argsort(order, dim=-1)


def inverse_permutation(perm: torch.Tensor) -> torch.Tensor:
    ranks = torch.arange(perm.shape[1]).expand_as(perm)  # ranks[b, k] = k
    return torch.zeros_like(perm).scatter(1, perm, ranks)


def keep_to_original_order(keep_sorted: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(keep_sorted).scatter(1, order, keep_sorted)
