# BUG: rank_descending reverses a stable ascending sort: the values come out largest first, but reversing also reverses the ties, so equal values are ranked highest index first.
import torch


def rank_descending(x: torch.Tensor) -> torch.Tensor:
    return torch.argsort(x, dim=-1, stable=True).flip(-1)


def inverse_permutation(perm: torch.Tensor) -> torch.Tensor:
    ranks = torch.arange(perm.shape[1]).expand_as(perm)  # ranks[b, k] = k
    return torch.zeros_like(perm).scatter(1, perm, ranks)


def keep_to_original_order(keep_sorted: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(keep_sorted).scatter(1, order, keep_sorted)
