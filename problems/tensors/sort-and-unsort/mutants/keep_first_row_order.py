# BUG: keep_to_original_order unsorts every row with the order of row 0; it is right only when all rows share one order.
import torch


def rank_descending(x: torch.Tensor) -> torch.Tensor:
    return torch.argsort(x, dim=-1, descending=True, stable=True)


def inverse_permutation(perm: torch.Tensor) -> torch.Tensor:
    ranks = torch.arange(perm.shape[1]).expand_as(perm)  # ranks[b, k] = k
    return torch.zeros_like(perm).scatter(1, perm, ranks)


def keep_to_original_order(keep_sorted: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(keep_sorted)
    out[:, order[0]] = keep_sorted
    return out
