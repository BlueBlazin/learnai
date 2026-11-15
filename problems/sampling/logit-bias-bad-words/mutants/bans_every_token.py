# BUG: when a banned sequence matches, removes every token of the sequence instead of only its last one.
import torch


def apply_bias(
    logits: torch.Tensor,
    bias: dict[int, float],
    allowed_token_ids: list[int] | None,
    bad_words: list[list[int]],
    history: list[int],
) -> torch.Tensor:
    vocab = logits.shape[0]
    delta = [0.0] * vocab
    for token, value in bias.items():
        delta[token] = value

    removed = [allowed_token_ids is not None] * vocab
    if allowed_token_ids is not None:
        for token in allowed_token_ids:
            removed[token] = False

    for word in bad_words:
        prefix = word[:-1]
        start = len(history) - len(prefix)
        if start >= 0 and history[start:] == prefix:
            for token in word:
                removed[token] = True

    biased = logits + torch.tensor(delta, dtype=logits.dtype)
    return torch.where(torch.tensor(removed), float("-inf"), biased)
