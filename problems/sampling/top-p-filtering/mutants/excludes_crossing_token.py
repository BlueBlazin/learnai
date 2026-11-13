# BUG: keeps tokens while the cumulative mass including the token is <= p, so the token that crosses p is dropped and the kept mass falls short of p.
import torch


def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    sorted_probs, order = torch.sort(probs, dim=-1, descending=True, stable=True)
    keep = torch.cumsum(sorted_probs, dim=-1) <= p
    keep[..., 0] = True  # at least one token survives
    kept = torch.where(keep, sorted_probs, 0.0)
    kept = kept / kept.sum(dim=-1, keepdim=True)
    return torch.zeros_like(probs).scatter(-1, order, kept)
