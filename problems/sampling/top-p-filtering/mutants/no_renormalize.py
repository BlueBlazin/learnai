# BUG: zeroes the removed tokens but never renormalizes, so each row sums to the kept mass instead of 1.
import torch


def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    sorted_probs, order = torch.sort(probs, dim=-1, descending=True, stable=True)
    mass_before = torch.cumsum(sorted_probs, dim=-1) - sorted_probs
    kept = torch.where(mass_before < p, sorted_probs, 0.0)
    return torch.zeros_like(probs).scatter(-1, order, kept)
