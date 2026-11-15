# BUG: groups tokens by expert separately for each of the k slots, so an expert that is first choice for some tokens and second choice for others is called twice.
import torch

from whetstone.kit import CountingExperts


def moe_sparse(
    x: torch.Tensor, w_router: torch.Tensor, experts: CountingExperts, k: int
) -> torch.Tensor:
    logits = x @ w_router.T
    chosen = torch.argsort(-logits, dim=-1, stable=True)[:, :k]
    weights = torch.softmax(torch.gather(logits, 1, chosen), dim=-1)

    y = torch.zeros_like(x)
    for slot in range(k):
        for expert in range(experts.n_experts):
            tokens = torch.nonzero(chosen[:, slot] == expert)[:, 0]
            if len(tokens) == 0:
                continue
            out = experts(expert, x[tokens])
            y = y.index_add(0, tokens, weights[tokens, slot, None] * out)
    return y
