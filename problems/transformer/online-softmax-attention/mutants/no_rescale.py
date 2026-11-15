# BUG: does not rescale the running sums when the running maximum grows, so terms from earlier blocks keep the weight they had relative to the old maximum.
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
        shift = torch.where(torch.isinf(new_top), 0.0, new_top)
        weights = torch.exp(scores - shift)
        norm = norm + weights.sum(dim=-1, keepdim=True)
        acc = acc + weights @ v
        top = new_top

    return acc / norm
