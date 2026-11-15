# BUG: subtracts the running maximum without a guard; while a query has seen no visible key the maximum is -inf, and -inf - (-inf) is NaN, which stays in its sums for good.
from collections.abc import Iterator

import torch

Block = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def blockwise_attention(q: torch.Tensor, kv_blocks: Iterator[Block]) -> torch.Tensor:
    n_queries, d = q.shape
    top = torch.full((n_queries, 1), float("-inf"))
    norm = torch.zeros(n_queries, 1)
    acc = 0.0

    for k, v, mask in kv_blocks:
        scores = torch.where(mask, (q @ k.T) * d**-0.5, float("-inf"))
        new_top = torch.maximum(top, scores.amax(dim=-1, keepdim=True))
        rescale = torch.exp(top - new_top)
        weights = torch.exp(scores - new_top)
        norm = norm * rescale + weights.sum(dim=-1, keepdim=True)
        acc = acc * rescale + weights @ v
        top = new_top

    return acc / norm
