# BUG: masks first and then writes logits[token] + bias for every biased token, which makes a removed token finite again.
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
            removed[word[-1]] = True

    out = torch.where(torch.tensor(removed), float("-inf"), logits)
    for token in bias:
        out[token] = logits[token] + delta[token]
    return out
