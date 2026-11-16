import torch

from learnai.kit import CountingExperts


def moe_sparse(
    x: torch.Tensor, w_router: torch.Tensor, experts: CountingExperts, k: int
) -> torch.Tensor:
    n_tokens = x.shape[0]
    # Route: the k best experts of every token, best first. The sort is stable, so
    # equal logits are ranked by expert index, lowest first.
    logits = x @ w_router.T  # (N, E)
    chosen = torch.argsort(-logits, dim=-1, stable=True)[:, :k]  # (N, k)
    # A softmax over the kept logits is the full softmax renormalized over them.
    weights = torch.softmax(torch.gather(logits, 1, chosen), dim=-1)  # (N, k)

    # Flatten to N * k (token, expert, weight) assignments and sort them by expert:
    # each expert's assignments become one contiguous run.
    flat_expert = chosen.reshape(-1)
    flat_token = torch.arange(n_tokens).repeat_interleave(k)
    flat_weight = weights.reshape(-1)
    order = torch.argsort(flat_expert, stable=True)
    counts = torch.bincount(flat_expert, minlength=experts.n_experts).tolist()

    y = torch.zeros_like(x)
    start = 0
    for expert, count in enumerate(counts):
        if count == 0:
            continue  # nobody chose this expert: no call at all
        run = order[start : start + count]
        start += count
        tokens = flat_token[run]  # the rows this expert must process, in any order
        out = experts(expert, x[tokens])  # one call, exactly its rows
        # Scatter back: row j of `out` belongs to token tokens[j]. A token never picks
        # the same expert twice, so the indices of one run are distinct.
        y = y.index_add(0, tokens, flat_weight[run, None] * out)
    return y
