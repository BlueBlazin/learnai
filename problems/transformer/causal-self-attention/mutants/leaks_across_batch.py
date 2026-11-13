# BUG: flattens the batch into one long sequence of B * T tokens before attending, so a batch item attends to the tokens of the items before it.
import math

import torch


def causal_self_attention(
    x: torch.Tensor,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    w_o: torch.Tensor,
) -> torch.Tensor:
    batch, seq, _ = x.shape
    x = x.reshape(batch * seq, -1)
    q, k, v = x @ w_q.T, x @ w_k.T, x @ w_v.T
    scores = q @ k.T / math.sqrt(q.shape[-1])
    future = torch.triu(torch.ones(batch * seq, batch * seq, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(future, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    return ((attn @ v) @ w_o.T).reshape(batch, seq, -1)
