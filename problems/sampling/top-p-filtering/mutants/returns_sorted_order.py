# BUG: returns the filtered probabilities in sorted (rank) order and never maps them back to token positions.
import torch


def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    sorted_probs, _ = torch.sort(probs, dim=-1, descending=True, stable=True)
    mass_before = torch.cumsum(sorted_probs, dim=-1) - sorted_probs
    kept = torch.where(mass_before < p, sorted_probs, 0.0)
    return kept / kept.sum(dim=-1, keepdim=True)
