# BUG: on rejection it resamples from the target distribution p instead of the residual max(0, p - q), which counts the mass that acceptance already delivered a second time.
import torch


def speculative_step(
    p: torch.Tensor, q: torch.Tensor, x: int, u_accept: float, u_resample: float
) -> tuple[int, bool]:
    if bool(u_accept * q[x] < p[x]):
        return x, True
    cdf = torch.cumsum(p, dim=0)
    token = int((cdf <= u_resample * cdf[-1]).sum())
    return token, False
