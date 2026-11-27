import torch


def rank_descending(x: torch.Tensor) -> torch.Tensor:
    # stable=True keeps equal values in index order, lowest first. Without it torch
    # makes no promise about the order of ties.
    return torch.argsort(x, dim=-1, descending=True, stable=True)


def inverse_permutation(perm: torch.Tensor) -> torch.Tensor:
    # perm[b, k] is where rank k points. The inverse answers the opposite question,
    # inv[b, perm[b, k]] = k, which is a scatter of the ranks 0 .. V - 1 through perm.
    ranks = torch.arange(perm.shape[1]).expand_as(perm)  # ranks[b, k] = k
    return torch.zeros_like(perm).scatter(1, perm, ranks)


def keep_to_original_order(keep_sorted: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
    # order[b, k] is the token at rank k, so the decision for rank k belongs at column
    # order[b, k]: a write through the permutation. A read through it would sort again.
    return torch.zeros_like(keep_sorted).scatter(1, order, keep_sorted)
