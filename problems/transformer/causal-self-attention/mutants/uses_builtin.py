# BUG: hands the projected queries, keys, and values to the built-in scaled_dot_product_attention, which the statement rules out.
import torch
import torch.nn.functional as F


def causal_self_attention(
    x: torch.Tensor,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    w_o: torch.Tensor,
) -> torch.Tensor:
    q, k, v = x @ w_q.T, x @ w_k.T, x @ w_v.T
    return F.scaled_dot_product_attention(q, k, v, is_causal=True) @ w_o.T
