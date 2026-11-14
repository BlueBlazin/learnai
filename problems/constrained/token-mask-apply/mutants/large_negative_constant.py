# BUG: disallowed logits are set to -1e9 instead of -inf, so the "removed" tokens still carry a finite logit that later arithmetic (a temperature, a bias, a finiteness test) can bring back.
import torch


def mask_logits(logits: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
    if not bool(allowed.any(dim=-1).all()):
        raise ValueError("a row of `allowed` has no allowed token")
    return torch.where(allowed, logits, -1e9)


def masked_probs(logits: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
    return torch.softmax(mask_logits(logits, allowed), dim=-1)
