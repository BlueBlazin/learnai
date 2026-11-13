# BUG: maps the filtered probabilities back with gather(order) instead of scatter(order), which applies the sort permutation a second time instead of inverting it.
import torch


def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    sorted_probs, order = torch.sort(probs, dim=-1, descending=True, stable=True)
    mass_before = torch.cumsum(sorted_probs, dim=-1) - sorted_probs
    kept = torch.where(mass_before < p, sorted_probs, 0.0)
    kept = kept / kept.sum(dim=-1, keepdim=True)
    return torch.gather(kept, -1, order)
