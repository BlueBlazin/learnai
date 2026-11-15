# BUG: normalizes inside each block and averages the per-block outputs, instead of carrying one normalizer across all blocks.
from collections.abc import Iterator

import torch

Block = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def blockwise_attention(q: torch.Tensor, kv_blocks: Iterator[Block]) -> torch.Tensor:
    d = q.shape[1]
    total = 0.0
    blocks = 0
    for k, v, mask in kv_blocks:
        scores = torch.where(mask, (q @ k.T) * d**-0.5, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        total = total + torch.nan_to_num(weights) @ v
        blocks += 1
    return total / blocks
