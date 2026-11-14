# BUG: counts with an indexed assignment (`counts[rows, tokens] += ...`), which does not accumulate repeated indices: a token generated several times is counted once.
import torch


def count_generated(
    tokens: torch.Tensor, prompt_lengths: torch.Tensor, vocab_size: int
) -> torch.Tensor:
    generated = torch.arange(tokens.shape[1]) >= prompt_lengths[:, None]
    rows = torch.arange(tokens.shape[0])[:, None]
    counts = torch.zeros(tokens.shape[0], vocab_size, dtype=torch.int64)
    counts[rows.expand_as(tokens)[generated], tokens[generated]] += 1
    return counts


def apply_penalties(
    logits: torch.Tensor,
    token_counts: torch.Tensor,
    frequency_penalty: float,
    presence_penalty: float,
) -> torch.Tensor:
    penalty = token_counts * frequency_penalty + (token_counts > 0) * presence_penalty
    return logits - penalty
