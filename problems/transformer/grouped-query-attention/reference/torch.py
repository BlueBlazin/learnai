import math

import torch


def gqa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # G query heads share each KV head: query head h reads KV head h // G.
    groups = q.shape[1] // k.shape[1]
    # Repeat every KV head G times in place, (B, H_kv, S, d) -> (B, H, S, d), so that
    # the head order is 0, 0, .., 1, 1, .. and lines up with the query heads.
    k = k.repeat_interleave(groups, dim=1)
    v = v.repeat_interleave(groups, dim=1)
    # From here on it is ordinary multi-head attention.
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])  # (B, H, T, S)
    scores = scores.masked_fill(~mask, float("-inf"))  # mask is True where attention is allowed
    return torch.softmax(scores, dim=-1) @ v
