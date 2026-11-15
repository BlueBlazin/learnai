# BUG: does not stop at the first accepted candidate: it keeps testing (against the same residual) and returns the last candidate that passes.
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
    accepted = None
    for i in range(n):
        x = candidates[i]
        # Accept with probability min(1, residual[x] / q[x]), without the division.
        if bool(u_accept[i] * q[x] < residual[x]):
            accepted = (int(x), i)
            continue
        # Rejected: what acceptance could not deliver is max(0, residual - q). Normalized,
        # it is the target of the next candidate, and of the final sample.
        residual = torch.clamp(residual - q, min=0.0)
        residual = residual / residual.sum()
    if accepted is not None:
        return accepted
    cdf = torch.cumsum(residual, dim=0)
    return int((cdf <= u_resample * cdf[-1]).sum()), n
