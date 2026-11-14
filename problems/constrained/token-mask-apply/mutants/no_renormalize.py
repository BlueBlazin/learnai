# BUG: masked_probs multiplies the unconstrained probabilities by the mask and does not renormalize, so each row sums to the allowed mass instead of 1.
import torch


def mask_logits(logits: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
    if not bool(allowed.any(dim=-1).all()):
        raise ValueError("a row of `allowed` has no allowed token")
    return torch.where(allowed, logits, float("-inf"))


def masked_probs(logits: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
    if not bool(allowed.any(dim=-1).all()):
        raise ValueError("a row of `allowed` has no allowed token")
    return torch.softmax(logits, dim=-1) * allowed
