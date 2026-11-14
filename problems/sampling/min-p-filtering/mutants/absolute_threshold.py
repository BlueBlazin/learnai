# BUG: uses min_p as an absolute probability cutoff instead of scaling it by the row's top probability; a flat row whose probabilities are all below min_p loses every token.
import torch


def min_p_filter(logits: torch.Tensor, min_p: float) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    return torch.where(probs >= min_p, logits, float("-inf"))
