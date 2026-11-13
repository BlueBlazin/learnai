# BUG: resamples from |p - q| instead of max(0, p - q), so tokens the draft model overrates (p < q) can be emitted on rejection.
import torch


def speculative_step(
    p: torch.Tensor, q: torch.Tensor, x: int, u_accept: float, u_resample: float
) -> tuple[int, bool]:
    if bool(u_accept * q[x] < p[x]):
        return x, True
    cdf = torch.cumsum(torch.abs(p - q), dim=0)
    token = int((cdf <= u_resample * cdf[-1]).sum())
    return token, False
