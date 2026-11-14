# BUG: stops when a full window no longer fits, so the tokens of a shorter last window are never scored, and divides by the number it did score.
from collections.abc import Callable

import torch


def perplexity(
    forward: Callable[[torch.Tensor], torch.Tensor],
    token_ids: torch.Tensor,
    max_len: int,
    stride: int,
) -> torch.Tensor:
    n = token_ids.shape[0]
    total = torch.zeros((), dtype=torch.float32)
    scored = 1  # tokens before this index are done; token 0 has no context to be scored from
    for begin in range(0, n, stride):
        end = min(begin + max_len, n)
        if end - begin < max_len and begin > 0:
            break
        logp = torch.log_softmax(forward(token_ids[begin:end]), dim=-1)  # (end - begin, V)
        # Row t of this window predicts token begin + t + 1. Only the tokens that no
        # earlier window scored are counted; the ones before them are context.
        rows = torch.arange(scored - 1 - begin, end - 1 - begin)
        total = total + logp[rows, token_ids[scored:end]].sum()
        scored = end
        if end == n:
            break
    return torch.exp(-total / (scored - 1))
