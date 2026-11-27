# BUG: rank_descending breaks ties by subtracting 1e-6 times the column from every value and sorting without stable=True: the bonus reorders values that differ by less than it, and near 100 float32 rounds it away.
import torch


def rank_descending(x: torch.Tensor) -> torch.Tensor:
    key = x - 1e-6 * torch.arange(x.shape[1])
    return torch.argsort(key, dim=-1, descending=True)


def inverse_permutation(perm: torch.Tensor) -> torch.Tensor:
    ranks = torch.arange(perm.shape[1]).expand_as(perm)  # ranks[b, k] = k
    return torch.zeros_like(perm).scatter(1, perm, ranks)


def keep_to_original_order(keep_sorted: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(keep_sorted).scatter(1, order, keep_sorted)
