# BUG: applies the penalties multiplicatively, like a repetition penalty (divide positive logits, multiply negative ones by 1 + the penalty), instead of subtracting them.
import torch


def count_generated(
    tokens: torch.Tensor, prompt_lengths: torch.Tensor, vocab_size: int
) -> torch.Tensor:
    generated = torch.arange(tokens.shape[1]) >= prompt_lengths[:, None]
    counts = torch.zeros(tokens.shape[0], vocab_size, dtype=torch.int64)
    return counts.scatter_add(1, tokens, generated.to(torch.int64))


def apply_penalties(
    logits: torch.Tensor,
    token_counts: torch.Tensor,
    frequency_penalty: float,
    presence_penalty: float,
) -> torch.Tensor:
    penalty = token_counts * frequency_penalty + (token_counts > 0) * presence_penalty
    factor = 1.0 + penalty
    return torch.where(logits > 0, logits / factor, logits * factor)
