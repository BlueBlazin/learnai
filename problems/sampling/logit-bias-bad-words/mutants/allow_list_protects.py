# BUG: applies the allow-list after the banned sequences and marks every allowed token as kept, so an allowed token that completes a banned sequence survives.
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

    removed = [False] * vocab

    for word in bad_words:
        prefix = word[:-1]
        start = len(history) - len(prefix)
        if start >= 0 and history[start:] == prefix:
            removed[word[-1]] = True

    if allowed_token_ids is not None:
        allowed = set(allowed_token_ids)
        removed = [token not in allowed for token in range(vocab)]

    biased = logits + torch.tensor(delta, dtype=logits.dtype)
    return torch.where(torch.tensor(removed), float("-inf"), biased)
