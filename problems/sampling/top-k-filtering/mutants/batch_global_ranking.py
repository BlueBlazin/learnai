# BUG: ranks the logits of the whole batch together and keeps the k largest overall, instead of the k largest of each row.
import torch


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 0 or k >= logits.shape[-1]:
        return logits
    order = torch.argsort(logits.flatten(), descending=True, stable=True)
    rank = torch.argsort(order).view_as(logits)
    return torch.where(rank < k, logits, float("-inf"))
