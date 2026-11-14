# BUG: treats the boolean mask as an additive one and adds it to the scores; allowed positions gain 1, and hidden positions are not hidden at all.
import math

import torch


def gqa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    groups = q.shape[1] // k.shape[1]
    k = k.repeat_interleave(groups, dim=1)
    v = v.repeat_interleave(groups, dim=1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    scores = scores + mask
    return torch.softmax(scores, dim=-1) @ v
