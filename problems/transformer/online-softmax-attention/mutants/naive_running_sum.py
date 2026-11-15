# BUG: accumulates exp(score) directly, with no running maximum; the sums overflow to inf for scores above 88 and underflow to 0 for scores below -103.
from collections.abc import Iterator

import torch

Block = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def blockwise_attention(q: torch.Tensor, kv_blocks: Iterator[Block]) -> torch.Tensor:
    n_queries, d = q.shape
    norm = torch.zeros(n_queries, 1)
    acc = 0.0
    for k, v, mask in kv_blocks:
        scores = torch.where(mask, (q @ k.T) * d**-0.5, float("-inf"))
        weights = torch.exp(scores)
        norm = norm + weights.sum(dim=-1, keepdim=True)
        acc = acc + weights @ v
    return acc / norm
