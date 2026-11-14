# BUG: sorts ascending and flips, which ranks equal logits by token index highest first, so the wrong members of a tie survive at the boundary.
import torch


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 0 or k >= logits.shape[-1]:
        return logits
    order = torch.argsort(logits, dim=-1, stable=True).flip(-1)
    rank = torch.argsort(order, dim=-1)
    return torch.where(rank < k, logits, float("-inf"))
