# BUG: shifts each block by its own maximum and never puts blocks on a common scale, so every block counts as if its best score were the best overall.
from collections.abc import Iterator

import torch

Block = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def blockwise_attention(q: torch.Tensor, kv_blocks: Iterator[Block]) -> torch.Tensor:
    n_queries, d = q.shape
    norm = torch.zeros(n_queries, 1)
    acc = 0.0
    for k, v, mask in kv_blocks:
        scores = torch.where(mask, (q @ k.T) * d**-0.5, float("-inf"))
        block_top = scores.amax(dim=-1, keepdim=True)
        shift = torch.where(torch.isinf(block_top), 0.0, block_top)
        weights = torch.exp(scores - shift)
        norm = norm + weights.sum(dim=-1, keepdim=True)
        acc = acc + weights @ v
    return acc / norm
