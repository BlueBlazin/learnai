# BUG: applies the penalty once per occurrence (penalty ** count) instead of once per distinct token.
import torch


def apply_repetition_penalty(
    logits: torch.Tensor, prev_tokens: torch.Tensor, penalty: float, pad_id: int
) -> torch.Tensor:
    real = (prev_tokens != pad_id).to(logits.dtype)
    counts = torch.zeros_like(logits).scatter_add(1, prev_tokens, real)
    factor = penalty**counts
    return torch.where(logits > 0, logits / factor, logits * factor)
