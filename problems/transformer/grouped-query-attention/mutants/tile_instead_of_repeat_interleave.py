# BUG: expands K and V with tile (head order 0, 1, 0, 1) instead of repeat_interleave (0, 0, 1, 1), so query head h reads KV head h % H_kv instead of h // (H / H_kv).
import math

import torch


def gqa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    groups = q.shape[1] // k.shape[1]
    k = k.tile(1, groups, 1, 1)
    v = v.tile(1, groups, 1, 1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    scores = scores.masked_fill(~mask, float("-inf"))
    return torch.softmax(scores, dim=-1) @ v
