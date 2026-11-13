import math

import torch


def causal_self_attention(
    x: torch.Tensor,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    w_o: torch.Tensor,
) -> torch.Tensor:
    seq = x.shape[1]
    # Weights are (out, in), so a projection is x @ W^T: (B, T, d) -> (B, T, out).
    q, k, v = x @ w_q.T, x @ w_k.T, x @ w_v.T
    # scores[b, i, j] = q[b, i] . k[b, j] / sqrt(d_k): row i is the query, column j the key.
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    # Position i may look at j <= i. Mask the logits (not the probabilities) so that
    # the softmax renormalizes over the visible positions only.
    future = torch.triu(torch.ones(seq, seq, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(future, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    return (attn @ v) @ w_o.T
