# BUG: accepts a draft token only when it is the target's argmax, which is greedy verification: the typical criterion is never applied.
import torch


def typical_accept(
    p: torch.Tensor, draft_tokens: torch.Tensor, epsilon: float, delta: float
) -> tuple[int, int]:
    k = draft_tokens.shape[0]
    accepted = (torch.argmax(p[:k], dim=-1) == draft_tokens).to(torch.int64)
    n_accepted = int(torch.cumprod(accepted, dim=0).sum())
    return n_accepted, int(torch.argmax(p[n_accepted]))
