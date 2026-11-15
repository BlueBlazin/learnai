# BUG: after a rejection only the rejected token is removed from the residual (set to zero, then renormalized) instead of subtracting the whole draft distribution q.
import torch


def multi_draft_step(
    p: torch.Tensor,
    q: torch.Tensor,
    candidates: torch.Tensor,
    u_accept: torch.Tensor,
    u_resample: float,
) -> tuple[int, int]:
    residual = p  # the distribution the next candidate has to deliver; sums to 1
    n = candidates.shape[0]
    for i in range(n):
        x = candidates[i]
        # Accept with probability min(1, residual[x] / q[x]), without the division.
        if bool(u_accept[i] * q[x] < residual[x]):
            return int(x), i
        # Rejected: what acceptance could not deliver is max(0, residual - q). Normalized,
        # it is the target of the next candidate, and of the final sample.
        residual = residual.clone()
        residual[x] = 0.0
        residual = residual / residual.sum()
    # Every candidate was rejected: sample the last residual by inverse CDF.
    cdf = torch.cumsum(residual, dim=0)
    return int((cdf <= u_resample * cdf[-1]).sum()), n
