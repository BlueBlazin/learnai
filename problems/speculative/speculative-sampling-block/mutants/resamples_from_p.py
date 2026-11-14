# BUG: after a rejection the replacement is sampled from the target distribution p[n] instead of the residual max(0, p[n] - q[n]), counting again the mass that acceptance already delivered.
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
    n_accepted = int(torch.cumprod(passes.to(torch.int64), dim=0).sum())
    cdf = torch.cumsum(p[n_accepted], dim=0)
    next_token = int((cdf <= u_resample * cdf[-1]).sum())
    return n_accepted, next_token
