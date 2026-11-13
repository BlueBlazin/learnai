# BUG: penalizes every token that occurs anywhere in the batch, in all rows, instead of using each row's own history.
import torch


def apply_repetition_penalty(
    logits: torch.Tensor, prev_tokens: torch.Tensor, penalty: float, pad_id: int
) -> torch.Tensor:
    seen = torch.zeros(logits.shape[1], dtype=torch.bool)
    seen[prev_tokens.flatten()] = True
    seen[pad_id] = False
    penalized = torch.where(logits > 0, logits / penalty, logits * penalty)
    return torch.where(seen, penalized, logits)
