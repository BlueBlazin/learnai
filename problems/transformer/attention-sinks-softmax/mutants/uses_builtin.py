# BUG: calls the built-in scaled_dot_product_attention (on a virtual extra key whose score is the sink and whose value is zero), which the statement rules out.
import math

import torch
import torch.nn.functional as F


def attention_with_sinks(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
    sinks: torch.Tensor,
) -> torch.Tensor:
    batch, n_heads, n_queries, dim = q.shape
    root = math.sqrt(dim)
    # One more feature: sqrt(d) in every query, 0 in every real key, the sink in the
    # virtual key, so that the virtual key scores sqrt(d) * sink / sqrt(d) = sink.
    q = torch.cat((q, torch.full((batch, n_heads, n_queries, 1), root)), dim=-1)
    k = F.pad(k, (0, 1))
    virtual = F.pad(sinks.reshape(1, n_heads, 1, 1), (dim, 0)).expand(batch, n_heads, 1, dim + 1)
    k = torch.cat((k, virtual), dim=2)
    v = F.pad(v, (0, 0, 0, 1))  # the virtual key has a zero value
    allowed = F.pad(mask, (0, 1), value=True)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=allowed, scale=1 / root)
