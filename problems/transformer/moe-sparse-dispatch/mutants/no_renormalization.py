# BUG: weights the experts by their probabilities under the softmax over all experts, without rescaling the k kept ones to sum to 1.
import torch

from learnai.kit import CountingExperts


def moe_sparse(
    x: torch.Tensor, w_router: torch.Tensor, experts: CountingExperts, k: int
) -> torch.Tensor:
    n_tokens = x.shape[0]
    logits = x @ w_router.T
    chosen = torch.argsort(-logits, dim=-1, stable=True)[:, :k]
    weights = torch.gather(torch.softmax(logits, dim=-1), 1, chosen)

    flat_expert = chosen.reshape(-1)
    flat_token = torch.arange(n_tokens).repeat_interleave(k)
    flat_weight = weights.reshape(-1)
    order = torch.argsort(flat_expert, stable=True)
    counts = torch.bincount(flat_expert, minlength=experts.n_experts).tolist()

    y = torch.zeros_like(x)
    start = 0
    for expert, count in enumerate(counts):
        if count == 0:
            continue
        run = order[start : start + count]
        start += count
        tokens = flat_token[run]
        out = experts(expert, x[tokens])
        y = y.index_add(0, tokens, flat_weight[run, None] * out)
    return y
