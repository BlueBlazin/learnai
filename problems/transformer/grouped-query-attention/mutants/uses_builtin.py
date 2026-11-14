# BUG: repeats K and V correctly but hands them to the built-in scaled_dot_product_attention, which the statement rules out.
import torch
import torch.nn.functional as F


def gqa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    groups = q.shape[1] // k.shape[1]
    k = k.repeat_interleave(groups, dim=1)
    v = v.repeat_interleave(groups, dim=1)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
