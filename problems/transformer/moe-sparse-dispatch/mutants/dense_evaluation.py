# BUG: runs every expert on every token and lets zero weights discard the rest; the output is right and the work is E / k times what the routing asks for.
import torch

from learnai.kit import CountingExperts


def moe_sparse(
    x: torch.Tensor, w_router: torch.Tensor, experts: CountingExperts, k: int
) -> torch.Tensor:
    logits = x @ w_router.T
    chosen = torch.argsort(-logits, dim=-1, stable=True)[:, :k]
    weights = torch.softmax(torch.gather(logits, 1, chosen), dim=-1)

    y = torch.zeros_like(x)
    for expert in range(experts.n_experts):
        share = torch.where(chosen == expert, weights, 0.0).sum(dim=-1, keepdim=True)
        y = y + share * experts(expert, x)
    return y
