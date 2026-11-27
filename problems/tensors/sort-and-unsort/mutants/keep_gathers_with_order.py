# BUG: keep_to_original_order indexes keep_sorted with order again (a gather) instead of writing through it: the decisions are permuted a second time, which is right only for a row whose order is its own inverse.
import torch


def rank_descending(x: torch.Tensor) -> torch.Tensor:
    return torch.argsort(x, dim=-1, descending=True, stable=True)


def inverse_permutation(perm: torch.Tensor) -> torch.Tensor:
    ranks = torch.arange(perm.shape[1]).expand_as(perm)  # ranks[b, k] = k
    return torch.zeros_like(perm).scatter(1, perm, ranks)


def keep_to_original_order(keep_sorted: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
    return torch.gather(keep_sorted, 1, order)
