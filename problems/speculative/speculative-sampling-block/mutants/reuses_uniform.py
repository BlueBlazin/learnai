# BUG: after a rejection it resamples with the uniform that decided the rejection, u_accept[n], instead of u_resample; given the rejection that number is no longer uniform on [0, 1).
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
    if n_accepted < k:
        weights = torch.clamp(p[n_accepted] - q[n_accepted], min=0.0)
        u_resample = float(u_accept[n_accepted])
    else:
        weights = p[k]
    cdf = torch.cumsum(weights, dim=0)
    next_token = int((cdf <= u_resample * cdf[-1]).sum())
    return n_accepted, next_token
