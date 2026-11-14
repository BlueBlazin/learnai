# BUG: a row with no allowed token is not detected; mask_logits returns a row of -inf and masked_probs a row of NaN instead of raising ValueError.
import torch


def mask_logits(logits: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
    return torch.where(allowed, logits, float("-inf"))


def masked_probs(logits: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
    return torch.softmax(mask_logits(logits, allowed), dim=-1)
