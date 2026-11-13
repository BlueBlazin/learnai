import torch


def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    # A stable descending sort ranks equal probabilities by token index, lowest first.
    sorted_probs, order = torch.sort(probs, dim=-1, descending=True, stable=True)
    # Mass of the tokens ranked strictly before each one. A token is kept while that
    # mass is still short of p, so the token that crosses p is kept, and rank 0 (mass
    # before it: exactly 0) always survives.
    mass_before = torch.cumsum(sorted_probs, dim=-1) - sorted_probs
    kept = torch.where(mass_before < p, sorted_probs, 0.0)
    kept = kept / kept.sum(dim=-1, keepdim=True)
    # `order[b, k]` is the token at rank k of row b: scatter sends rank k back there.
    return torch.zeros_like(probs).scatter(-1, order, kept)
