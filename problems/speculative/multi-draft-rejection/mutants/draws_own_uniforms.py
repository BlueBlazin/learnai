# BUG: draws its own uniform numbers with torch.rand instead of using u_accept and u_resample, which the statement rules out; the results are not reproducible.
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
        if bool(torch.rand(()) * q[x] < residual[x]):
            return int(x), i
        # Rejected: what acceptance could not deliver is max(0, residual - q). Normalized,
        # it is the target of the next candidate, and of the final sample.
        residual = torch.clamp(residual - q, min=0.0)
        residual = residual / residual.sum()
    # Every candidate was rejected: sample the last residual by inverse CDF.
    cdf = torch.cumsum(residual, dim=0)
    return int((cdf <= torch.rand(()) * cdf[-1]).sum()), n
