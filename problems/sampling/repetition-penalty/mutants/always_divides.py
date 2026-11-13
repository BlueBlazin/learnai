# BUG: divides every seen logit by the penalty, which makes negative logits less negative and so raises the probability of those tokens.
import torch


def apply_repetition_penalty(
    logits: torch.Tensor, prev_tokens: torch.Tensor, penalty: float, pad_id: int
) -> torch.Tensor:
    seen = torch.zeros_like(logits, dtype=torch.bool).scatter(1, prev_tokens, prev_tokens != pad_id)
    return torch.where(seen, logits / penalty, logits)
