# BUG: ignores stride and cuts the text into disjoint chunks of max_len tokens: the first token of every later chunk is never scored and the tokens after it get almost no context.
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
    count = 0
    for begin in range(0, n, max_len):
        end = min(begin + max_len, n)
        logp = torch.log_softmax(forward(token_ids[begin:end]), dim=-1)  # (end - begin, V)
        # Row t of this window predicts token begin + t + 1. Only the tokens that no
        # earlier window scored are counted; the ones before them are context.
        rows = torch.arange(0, end - 1 - begin)
        total = total + logp[rows, token_ids[begin + 1 : end]].sum()
        count += end - 1 - begin
        if end == n:
            break
    return torch.exp(-total / count)
