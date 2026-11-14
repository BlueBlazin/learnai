# BUG: divides the scores by sqrt(d * H / H_kv), as if a shared KV head were G times wider; the scale is sqrt(d) whatever the grouping.
import math

import torch


def gqa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    groups = q.shape[1] // k.shape[1]
    k = k.repeat_interleave(groups, dim=1)
    v = v.repeat_interleave(groups, dim=1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1] * groups)
    scores = scores.masked_fill(~mask, float("-inf"))
    return torch.softmax(scores, dim=-1) @ v
