# BUG: looks for the prefix anywhere in the history instead of at its end, so a sequence stays banned long after the context has moved on.
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
        size = len(prefix)
        if any(history[i : i + size] == prefix for i in range(len(history) - size + 1)):
            removed[word[-1]] = True

    biased = logits + torch.tensor(delta, dtype=logits.dtype)
    return torch.where(torch.tensor(removed), float("-inf"), biased)
