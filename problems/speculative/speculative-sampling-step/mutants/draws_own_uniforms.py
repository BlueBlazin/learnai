# BUG: draws its own uniform numbers with torch.rand instead of using u_accept and u_resample, which the statement rules out; the results are not reproducible.
import torch


def speculative_step(
    p: torch.Tensor, q: torch.Tensor, x: int, u_accept: float, u_resample: float
) -> tuple[int, bool]:
    if bool(torch.rand(()) * q[x] < p[x]):
        return x, True
    cdf = torch.cumsum(torch.clamp(p - q, min=0.0), dim=0)
    token = int((cdf <= torch.rand(()) * cdf[-1]).sum())
    return token, False
