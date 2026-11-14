# BUG: keeps tokens with probability strictly greater than the cutoff; with min_p = 1 the top token equals the cutoff and is removed, leaving a row with no token.
import torch


def min_p_filter(logits: torch.Tensor, min_p: float) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    threshold = min_p * torch.amax(probs, dim=-1, keepdim=True)
    return torch.where(probs > threshold, logits, float("-inf"))
