# BUG: masks removed entries with the most negative finite float32 instead of -inf, so they are not masked: later filters count them and softmax can give them weight.
import torch


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 0 or k >= logits.shape[-1]:
        return logits
    order = torch.argsort(logits, dim=-1, descending=True, stable=True)
    rank = torch.argsort(order, dim=-1)
    return torch.where(rank < k, logits, torch.finfo(logits.dtype).min)
