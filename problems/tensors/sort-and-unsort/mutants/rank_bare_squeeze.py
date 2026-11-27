# BUG: rank_descending ends in a bare squeeze(), which removes the batch axis of a single row and the column axis when V == 1.
import torch


def rank_descending(x: torch.Tensor) -> torch.Tensor:
    return torch.sort(x, dim=-1, descending=True, stable=True).indices.squeeze()


def inverse_permutation(perm: torch.Tensor) -> torch.Tensor:
    ranks = torch.arange(perm.shape[1]).expand_as(perm)  # ranks[b, k] = k
    return torch.zeros_like(perm).scatter(1, perm, ranks)


def keep_to_original_order(keep_sorted: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(keep_sorted).scatter(1, order, keep_sorted)
