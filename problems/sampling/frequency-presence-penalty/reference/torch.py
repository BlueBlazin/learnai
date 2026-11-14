import torch


def count_generated(
    tokens: torch.Tensor, prompt_lengths: torch.Tensor, vocab_size: int
) -> torch.Tensor:
    # generated[b, t]: is position t of row b past that row's prompt?
    generated = torch.arange(tokens.shape[1]) >= prompt_lengths[:, None]
    # A histogram per row: add 1 at column tokens[b, t] for every generated position and
    # 0 for every prompt position. scatter_add accumulates repeated indices, which is
    # the point: a token generated three times must count three.
    counts = torch.zeros(tokens.shape[0], vocab_size, dtype=torch.int64)
    return counts.scatter_add(1, tokens, generated.to(torch.int64))


def apply_penalties(
    logits: torch.Tensor,
    token_counts: torch.Tensor,
    frequency_penalty: float,
    presence_penalty: float,
) -> torch.Tensor:
    # Frequency grows with every occurrence; presence is charged once, as soon as the
    # token has occurred at all. Both are subtracted, whatever the sign of the logit.
    penalty = token_counts * frequency_penalty + (token_counts > 0) * presence_penalty
    return logits - penalty
