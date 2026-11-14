# BUG: maps query head h to KV head h // H_kv instead of h // (H / H_kv); the two agree only when H = H_kv * H_kv, and with a single KV head every query head but the first points past it.
import math

import torch


def gqa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    kv_head = torch.arange(q.shape[1]) // k.shape[1]
    k = k[:, kv_head]
    v = v[:, kv_head]
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    scores = scores.masked_fill(~mask, float("-inf"))
    return torch.softmax(scores, dim=-1) @ v
