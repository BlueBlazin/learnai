# BUG: shifts the removal mask right by one although the cumulative sum already excludes the token itself, so one token past the crossing token is kept.
import torch


def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    sorted_probs, order = torch.sort(probs, dim=-1, descending=True, stable=True)
    mass_before = torch.cumsum(sorted_probs, dim=-1) - sorted_probs
    remove = mass_before >= p
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False
    kept = torch.where(remove, 0.0, sorted_probs)
    kept = kept / kept.sum(dim=-1, keepdim=True)
    return torch.zeros_like(probs).scatter(-1, order, kept)
