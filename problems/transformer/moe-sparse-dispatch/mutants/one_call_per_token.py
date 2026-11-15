# BUG: calls an expert once per token and slot with a single row, k * N calls, instead of once per expert with all of its rows.
import torch

from whetstone.kit import CountingExperts


def moe_sparse(
    x: torch.Tensor, w_router: torch.Tensor, experts: CountingExperts, k: int
) -> torch.Tensor:
    n_tokens = x.shape[0]
    logits = x @ w_router.T
    chosen = torch.argsort(-logits, dim=-1, stable=True)[:, :k]
    weights = torch.softmax(torch.gather(logits, 1, chosen), dim=-1)

    y = torch.zeros_like(x)
    for token in range(n_tokens):
        for slot in range(k):
            out = experts(int(chosen[token, slot]), x[token : token + 1])
            y[token] += weights[token, slot] * out[0]
    return y
