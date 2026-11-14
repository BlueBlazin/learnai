# BUG: the first row's mask is applied to every row of the batch, as if all sequences were in the same constraint state.
import torch


def mask_logits(logits: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
    if not bool(allowed.any(dim=-1).all()):
        raise ValueError("a row of `allowed` has no allowed token")
    return torch.where(allowed[0], logits, float("-inf"))


def masked_probs(logits: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
    return torch.softmax(mask_logits(logits, allowed), dim=-1)
