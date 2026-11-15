# BUG: the sink logit is added to every score of its head instead of being a column of its own; a softmax ignores a constant shift, so the sink has no effect.
import math

import torch


def attention_with_sinks(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
    sinks: torch.Tensor,
) -> torch.Tensor:
    n_heads = q.shape[1]
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])  # (B, H, T, S)
    scores = scores.masked_fill(~mask, float("-inf"))  # mask is True where attention is allowed
    # One extra column per row holds the head's sink logit. It is not scaled by
    # 1/sqrt(d) and never masked: it is a logit, not a score against a key.
    scores = scores + sinks.reshape(1, n_heads, 1, 1)
    return torch.softmax(scores, dim=-1) @ v
