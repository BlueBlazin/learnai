import torch


def min_p_filter(logits: torch.Tensor, min_p: float) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    # The cutoff is relative to each row's own top probability, so it adapts: a confident
    # row gets a high cutoff and keeps few tokens, a flat row gets a low one and keeps many.
    threshold = min_p * torch.amax(probs, dim=-1, keepdim=True)
    # `>=` keeps the top token itself (and its ties) for every min_p <= 1, and keeps
    # everything when min_p is 0. Removed tokens are masked in logit space.
    return torch.where(probs >= threshold, logits, float("-inf"))
