# BUG: after K acceptances the bonus token is sampled from p[K - 1], the row of the last draft token, instead of the extra row p[K].
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
    else:
        weights = p[k - 1]
    cdf = torch.cumsum(weights, dim=0)
    next_token = int((cdf <= u_resample * cdf[-1]).sum())
    return n_accepted, next_token
