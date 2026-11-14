import torch


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 0 or k >= logits.shape[-1]:
        return logits  # k = 0 means "top-k is off"; k >= V keeps everything
    # A stable descending sort ranks equal logits by token index, lowest first:
    # order[b, r] is the token at rank r of row b.
    order = torch.argsort(logits, dim=-1, descending=True, stable=True)
    # The argsort of a permutation is its inverse: rank[b, v] is the rank of token v.
    rank = torch.argsort(order, dim=-1)
    # Ranks are distinct, so exactly k entries per row have rank < k.
    return torch.where(rank < k, logits, float("-inf"))
