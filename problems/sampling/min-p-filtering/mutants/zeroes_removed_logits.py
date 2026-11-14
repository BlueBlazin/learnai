# BUG: sets removed entries to 0 instead of -inf; 0 is a removed probability, but as a logit it is an ordinary, often large, value.
import torch


def min_p_filter(logits: torch.Tensor, min_p: float) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    threshold = min_p * torch.amax(probs, dim=-1, keepdim=True)
    return torch.where(probs >= threshold, logits, 0.0)
