# BUG: takes the top probability over the whole batch (max without an axis) instead of per row, so one confident row raises the cutoff of every other row.
import torch


def min_p_filter(logits: torch.Tensor, min_p: float) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    threshold = min_p * probs.max()
    return torch.where(probs >= threshold, logits, float("-inf"))
