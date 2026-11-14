import torch


def mask_logits(logits: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
    # A row with no allowed token has no valid next token: fail loudly instead of
    # handing an all -inf row to softmax, which would turn it into NaN.
    if not bool(allowed.any(dim=-1).all()):
        raise ValueError("a row of `allowed` has no allowed token")
    # -inf, not a large negative number: exp(-inf) is exactly 0 for any other logits.
    return torch.where(allowed, logits, float("-inf"))


def masked_probs(logits: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
    # Mask first, normalize second: the softmax then runs over the allowed tokens only,
    # and its shift by the row maximum uses the largest *allowed* logit.
    return torch.softmax(mask_logits(logits, allowed), dim=-1)
