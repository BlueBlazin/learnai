# BUG: the empty-mask check looks at the whole batch (`allowed.any()`) instead of every row, so one dead row next to live ones is not reported and comes out as NaN.
import torch


def mask_logits(logits: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
    if not bool(allowed.any()):
        raise ValueError("a row of `allowed` has no allowed token")
    return torch.where(allowed, logits, float("-inf"))


def masked_probs(logits: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
    return torch.softmax(mask_logits(logits, allowed), dim=-1)
