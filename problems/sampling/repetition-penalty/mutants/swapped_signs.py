# BUG: multiplies positive logits and divides negative ones, the reverse of the rule, so repeated tokens are rewarded.
import torch


def apply_repetition_penalty(
    logits: torch.Tensor, prev_tokens: torch.Tensor, penalty: float, pad_id: int
) -> torch.Tensor:
    seen = torch.zeros_like(logits, dtype=torch.bool).scatter(1, prev_tokens, prev_tokens != pad_id)
    penalized = torch.where(logits > 0, logits * penalty, logits / penalty)
    return torch.where(seen, penalized, logits)
