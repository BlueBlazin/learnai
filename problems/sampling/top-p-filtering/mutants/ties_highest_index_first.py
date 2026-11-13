# BUG: sorts ascending and flips, which ranks equal probabilities by token index highest first, so the wrong members of a tie survive at the boundary.
import torch


def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    sorted_probs, order = torch.sort(probs, dim=-1, stable=True)
    sorted_probs, order = sorted_probs.flip(-1), order.flip(-1)
    mass_before = torch.cumsum(sorted_probs, dim=-1) - sorted_probs
    kept = torch.where(mass_before < p, sorted_probs, 0.0)
    kept = kept / kept.sum(dim=-1, keepdim=True)
    return torch.zeros_like(probs).scatter(-1, order, kept)
