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
    # Draft i passes its own test when u_i < p_i[x_i] / q_i[x_i]; multiplying through
    # by q avoids the division and the min(1, .).
    passes = u_accept * q[rows, draft_tokens] < p[rows, draft_tokens]
    # Only the prefix before the first failure is accepted: the running product is 1
    # up to the first failed test and 0 afterwards, whatever the later tests say.
    n_accepted = int(torch.cumprod(passes.to(torch.int64), dim=0).sum())
    if n_accepted < k:
        # Rejected at position n: the replacement comes from the residual of that
        # position, max(0, p_n - q_n), normalized.
        weights = torch.clamp(p[n_accepted] - q[n_accepted], min=0.0)
    else:
        # Everything accepted: nothing to correct, so the bonus token is an ordinary
        # sample from the target's distribution in the extra row.
        weights = p[k]
    # Inverse CDF with u_resample. Scaling the uniform by the total normalizes the
    # weights; the first index whose cumulative sum exceeds the threshold equals the
    # number of entries that do not exceed it.
    cdf = torch.cumsum(weights, dim=0)
    next_token = int((cdf <= u_resample * cdf[-1]).sum())
    return n_accepted, next_token
