# BUG: averages the queries of each group into one query per KV head, attends once per KV head, and copies the result to the group's heads; the heads of a group must stay distinct.
import math

import torch


def gqa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    batch, heads, seq, dim = q.shape
    groups = heads // k.shape[1]
    q = q.reshape(batch, k.shape[1], groups, seq, dim).mean(dim=2)
    scores = q @ k.transpose(-2, -1) / math.sqrt(dim)
    scores = scores.masked_fill(~mask, float("-inf"))
    out = torch.softmax(scores, dim=-1) @ v
    return out.repeat_interleave(groups, dim=1)
