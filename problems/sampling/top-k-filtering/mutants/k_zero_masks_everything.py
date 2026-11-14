# BUG: has no special case for k = 0, so "top-k disabled" keeps zero tokens and every row comes back entirely -inf.
import torch


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    order = torch.argsort(logits, dim=-1, descending=True, stable=True)
    rank = torch.argsort(order, dim=-1)
    return torch.where(rank < k, logits, float("-inf"))
