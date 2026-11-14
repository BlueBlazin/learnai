# BUG: applies the ratio to logits instead of probabilities (logit >= min_p * top logit); a ratio of probabilities is a difference of logits, not a ratio of logits.
import torch


def min_p_filter(logits: torch.Tensor, min_p: float) -> torch.Tensor:
    threshold = min_p * torch.amax(logits, dim=-1, keepdim=True)
    return torch.where(logits >= threshold, logits, float("-inf"))
