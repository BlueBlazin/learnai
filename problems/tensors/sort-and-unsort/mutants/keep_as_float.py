# BUG: keep_to_original_order scatters into a float32 tensor and returns zeros and ones instead of a boolean tensor.
import torch


def rank_descending(x: torch.Tensor) -> torch.Tensor:
    return torch.argsort(x, dim=-1, descending=True, stable=True)


def inverse_permutation(perm: torch.Tensor) -> torch.Tensor:
    ranks = torch.arange(perm.shape[1]).expand_as(perm)  # ranks[b, k] = k
    return torch.zeros_like(perm).scatter(1, perm, ranks)


def keep_to_original_order(keep_sorted: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
    out = torch.zeros(keep_sorted.shape, dtype=torch.float32)
    return out.scatter(1, order, keep_sorted.to(torch.float32))
