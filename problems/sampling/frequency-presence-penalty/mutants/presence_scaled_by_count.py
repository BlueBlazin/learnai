# BUG: multiplies the presence penalty by the count as well, so it is charged per occurrence instead of once and the two penalties collapse into one.
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
    penalty = token_counts * frequency_penalty + token_counts * presence_penalty
    return logits - penalty
