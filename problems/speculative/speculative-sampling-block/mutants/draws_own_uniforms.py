# BUG: draws its own uniform numbers with torch.rand instead of using u_accept and u_resample, which the statement rules out; the results are not reproducible.
import torch


def verify_block(
    p: torch.Tensor,
    q: torch.Tensor,
    draft_tokens: torch.Tensor,
    u_accept: torch.Tensor,
    u_resample: float,
) -> tuple[int, int]:
    k = q.shape[0]
    rows = torch.arange(k)
    passes = torch.rand(k) * q[rows, draft_tokens] < p[rows, draft_tokens]
    n_accepted = int(torch.cumprod(passes.to(torch.int64), dim=0).sum())
    if n_accepted < k:
        weights = torch.clamp(p[n_accepted] - q[n_accepted], min=0.0)
    else:
        weights = p[k]
    cdf = torch.cumsum(weights, dim=0)
    next_token = int((cdf <= torch.rand(()) * cdf[-1]).sum())
    return n_accepted, next_token
