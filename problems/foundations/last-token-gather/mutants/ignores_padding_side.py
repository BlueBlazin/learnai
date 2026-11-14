# BUG: always uses position lengths[b] - 1, which is right for right padding and points into the middle of the row (or into the padding) for left padding.
import torch


def last_token_logits(
    logits: torch.Tensor, lengths: torch.Tensor, padding_side: str
) -> torch.Tensor:
    rows = torch.arange(logits.shape[0], device=logits.device)
    return logits[rows, lengths - 1]
