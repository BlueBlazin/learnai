# BUG: the standard deviation is taken over the whole row; a masked -inf entry makes it NaN, every comparison is false, and the row is emptied.
import torch


def top_n_sigma_filter(logits: torch.Tensor, n: float) -> torch.Tensor:
    if n <= 0:
        return logits
    std = torch.std(logits, dim=-1, keepdim=True, correction=0)
    cutoff = torch.amax(logits, dim=-1, keepdim=True) - n * std
    return torch.where(logits >= cutoff, logits, float("-inf"))
