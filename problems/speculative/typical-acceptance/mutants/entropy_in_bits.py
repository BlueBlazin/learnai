# BUG: the entropy is computed with log2 (bits) instead of the natural logarithm, so exp(-H) is too small and uncertain rows accept too much.
import torch


def typical_accept(
    p: torch.Tensor, draft_tokens: torch.Tensor, epsilon: float, delta: float
) -> tuple[int, int]:
    k = draft_tokens.shape[0]
    entropy = -(p * torch.log2(torch.where(p > 0, p, torch.ones_like(p)))).sum(dim=-1)
    threshold = torch.clamp(delta * torch.exp(-entropy), max=epsilon)
    prob = p[torch.arange(k), draft_tokens]
    accepted = (prob > threshold[:k]).to(torch.int64)
    n_accepted = int(torch.cumprod(accepted, dim=0).sum())
    return n_accepted, int(torch.argmax(p[n_accepted]))
