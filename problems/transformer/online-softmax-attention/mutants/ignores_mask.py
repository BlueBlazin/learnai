# BUG: never applies the mask: every query attends to every key of every block.
from collections.abc import Iterator

import torch

Block = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def blockwise_attention(q: torch.Tensor, kv_blocks: Iterator[Block]) -> torch.Tensor:
    n_queries, d = q.shape
    top = torch.full((n_queries, 1), float("-inf"))
    norm = torch.zeros(n_queries, 1)
    acc = 0.0

    for k, v, _ in kv_blocks:
        scores = (q @ k.T) * d**-0.5
        new_top = torch.maximum(top, scores.amax(dim=-1, keepdim=True))
        shift = torch.where(torch.isinf(new_top), 0.0, new_top)
        rescale = torch.exp(top - shift)
        weights = torch.exp(scores - shift)
        norm = norm * rescale + weights.sum(dim=-1, keepdim=True)
        acc = acc * rescale + weights @ v
        top = new_top

    return acc / norm
