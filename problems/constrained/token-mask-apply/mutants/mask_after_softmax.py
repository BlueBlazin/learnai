# BUG: masked_probs takes the softmax over all tokens, zeroes the disallowed ones, and divides by what is left; when the allowed tokens are far below a disallowed one their probabilities have already underflowed to 0 and the row becomes 0 / 0.
import torch


def mask_logits(logits: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
    if not bool(allowed.any(dim=-1).all()):
        raise ValueError("a row of `allowed` has no allowed token")
    return torch.where(allowed, logits, float("-inf"))


def masked_probs(logits: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
    if not bool(allowed.any(dim=-1).all()):
        raise ValueError("a row of `allowed` has no allowed token")
    kept = torch.softmax(logits, dim=-1) * allowed
    return kept / kept.sum(dim=-1, keepdim=True)
