# BUG: every candidate is tested against the original p, and the final sample comes from the first residual max(0, p - q): the residual is not updated after a rejection.
import torch


def multi_draft_step(
    p: torch.Tensor,
    q: torch.Tensor,
    candidates: torch.Tensor,
    u_accept: torch.Tensor,
    u_resample: float,
) -> tuple[int, int]:
    residual = p
    first = torch.clamp(p - q, min=0.0)
    n = candidates.shape[0]
    for i in range(n):
        x = candidates[i]
        # Accept with probability min(1, residual[x] / q[x]), without the division.
        if bool(u_accept[i] * q[x] < residual[x]):
            return int(x), i
        # Rejected: what acceptance could not deliver is max(0, residual - q). Normalized,
        # it is the target of the next candidate, and of the final sample.
    # Every candidate was rejected: sample the last residual by inverse CDF.
    cdf = torch.cumsum(first, dim=0)
    return int((cdf <= u_resample * cdf[-1]).sum()), n
