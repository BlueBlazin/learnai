from collections.abc import Callable

import torch


def perplexity(
    forward: Callable[[torch.Tensor], torch.Tensor],
    token_ids: torch.Tensor,
    max_len: int,
    stride: int,
) -> torch.Tensor:
    """Perplexity of `token_ids` under a model that reads at most `max_len` tokens.

    forward: forward(ids) -> logits. ids: integer tensor of shape (T,), 1 <= T <= max_len
        (more raises ValueError). logits: float32, shape (T, V); row t scores the token
        that follows ids[t], given ids[0..t] and nothing else.
    token_ids: integer tensor of shape (N,), N >= 2.
    max_len: the longest window `forward` accepts.
    stride: how far each window starts after the previous one, 1 <= stride < max_len.
    Returns a 0-d float32 tensor: exp of the mean negative natural-log probability of
    tokens 1 .. N - 1, each scored once, by the first window that can predict it.
    """
    raise NotImplementedError
