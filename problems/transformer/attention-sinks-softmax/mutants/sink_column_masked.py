# BUG: the mask is applied after the sink column is appended, with the mask padded by False for that column, so the sink is masked out and has no effect.
import math

import torch


def attention_with_sinks(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
    sinks: torch.Tensor,
) -> torch.Tensor:
    batch, n_heads, n_queries, _ = q.shape
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])  # (B, H, T, S)
    # One extra column per row holds the head's sink logit. It is not scaled by
    # 1/sqrt(d) and never masked: it is a logit, not a score against a key.
    sink = sinks.reshape(1, n_heads, 1, 1).expand(batch, n_heads, n_queries, 1)
    logits = torch.cat((scores, sink), dim=-1)
    allowed = torch.nn.functional.pad(mask, (0, 1), value=False)  # (T, S + 1)
    weights = torch.softmax(logits.masked_fill(~allowed, float("-inf")), dim=-1)
    # The sink takes its share of the probability and has no value: drop its column.
    return weights[..., :-1] @ v
