# BUG: counts every position of the row, prompt included, although only generated tokens are penalized.
import torch


def count_generated(
    tokens: torch.Tensor, prompt_lengths: torch.Tensor, vocab_size: int
) -> torch.Tensor:
    counts = torch.zeros(tokens.shape[0], vocab_size, dtype=torch.int64)
    return counts.scatter_add(1, tokens, torch.ones_like(tokens))


def apply_penalties(
    logits: torch.Tensor,
    token_counts: torch.Tensor,
    frequency_penalty: float,
    presence_penalty: float,
) -> torch.Tensor:
    penalty = token_counts * frequency_penalty + (token_counts > 0) * presence_penalty
    return logits - penalty
