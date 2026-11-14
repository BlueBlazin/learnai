# BUG: off by one in the rank comparison (`<=` for `<`): keeps the k + 1 largest entries of every row.
import torch


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 0 or k >= logits.shape[-1]:
        return logits
    order = torch.argsort(logits, dim=-1, descending=True, stable=True)
    rank = torch.argsort(order, dim=-1)
    return torch.where(rank <= k, logits, float("-inf"))
