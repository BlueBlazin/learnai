# BUG: collects every block and then computes ordinary attention over the concatenation; the numbers are right, and all keys and values are held at once.
from collections.abc import Iterator

import torch

Block = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def blockwise_attention(q: torch.Tensor, kv_blocks: Iterator[Block]) -> torch.Tensor:
    d = q.shape[1]
    blocks = list(kv_blocks)
    k = torch.cat([block[0] for block in blocks])
    v = torch.cat([block[1] for block in blocks])
    mask = torch.cat([block[2] for block in blocks], dim=1)
    scores = torch.where(mask, (q @ k.T) * d**-0.5, float("-inf"))
    return torch.softmax(scores, dim=-1) @ v
