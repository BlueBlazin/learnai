# BUG: turns the cutoff into one probability threshold for the whole batch (the smallest kept probability of any row) instead of one cutoff per row.
import torch


def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    sorted_probs, _ = torch.sort(probs, dim=-1, descending=True, stable=True)
    mass_before = torch.cumsum(sorted_probs, dim=-1) - sorted_probs
    threshold = sorted_probs[mass_before < p].min()
    kept = torch.where(probs >= threshold, probs, 0.0)
    return kept / kept.sum(dim=-1, keepdim=True)
