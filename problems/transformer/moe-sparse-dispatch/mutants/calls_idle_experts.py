# BUG: calls every expert, also those that no token selected, with an empty batch of rows; each of those is a wasted call.
import torch

from whetstone.kit import CountingExperts


def moe_sparse(
    x: torch.Tensor, w_router: torch.Tensor, experts: CountingExperts, k: int
) -> torch.Tensor:
    n_tokens = x.shape[0]
    logits = x @ w_router.T
    chosen = torch.argsort(-logits, dim=-1, stable=True)[:, :k]
    weights = torch.softmax(torch.gather(logits, 1, chosen), dim=-1)

    flat_expert = chosen.reshape(-1)
    flat_token = torch.arange(n_tokens).repeat_interleave(k)
    flat_weight = weights.reshape(-1)
    order = torch.argsort(flat_expert, stable=True)
    counts = torch.bincount(flat_expert, minlength=experts.n_experts).tolist()

    y = torch.zeros_like(x)
    start = 0
    for expert, count in enumerate(counts):
        run = order[start : start + count]
        start += count
        tokens = flat_token[run]
        out = experts(expert, x[tokens])
        y = y.index_add(0, tokens, flat_weight[run, None] * out)
    return y
