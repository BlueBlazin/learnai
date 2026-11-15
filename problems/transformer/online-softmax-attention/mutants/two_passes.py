# BUG: finds the maximum in a first pass over the blocks and sums in a second one, but the iterator is one-pass: the second loop receives nothing.
from collections.abc import Iterator

import torch

Block = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def blockwise_attention(q: torch.Tensor, kv_blocks: Iterator[Block]) -> torch.Tensor:
    n_queries, d = q.shape
    top = torch.full((n_queries, 1), float("-inf"))
    for k, _, mask in kv_blocks:
        scores = torch.where(mask, (q @ k.T) * d**-0.5, float("-inf"))
        top = torch.maximum(top, scores.amax(dim=-1, keepdim=True))

    norm = torch.zeros(n_queries, 1)
    acc = torch.zeros(n_queries, 1)
    for k, v, mask in kv_blocks:
        scores = torch.where(mask, (q @ k.T) * d**-0.5, float("-inf"))
        weights = torch.exp(scores - top)
        norm = norm + weights.sum(dim=-1, keepdim=True)
        acc = acc + weights @ v
    return acc / norm
