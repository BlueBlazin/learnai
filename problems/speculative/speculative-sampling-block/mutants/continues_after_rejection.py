# BUG: counts every draft that passes its own test, so drafts after the first rejection are accepted although they were sampled from a context that contains the rejected token.
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
    passes = u_accept * q[rows, draft_tokens] < p[rows, draft_tokens]
    n_accepted = int(passes.sum())
    if n_accepted < k:
        weights = torch.clamp(p[n_accepted] - q[n_accepted], min=0.0)
    else:
        weights = p[k]
    cdf = torch.cumsum(weights, dim=0)
    next_token = int((cdf <= u_resample * cdf[-1]).sum())
    return n_accepted, next_token
