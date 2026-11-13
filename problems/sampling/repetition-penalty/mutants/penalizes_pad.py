# BUG: treats padding entries as real history, so the logit of pad_id is penalized in every row that has padding.
import torch


def apply_repetition_penalty(
    logits: torch.Tensor, prev_tokens: torch.Tensor, penalty: float, pad_id: int
) -> torch.Tensor:
    seen = torch.zeros_like(logits, dtype=torch.bool).scatter(1, prev_tokens, True)
    penalized = torch.where(logits > 0, logits / penalty, logits * penalty)
    return torch.where(seen, penalized, logits)
