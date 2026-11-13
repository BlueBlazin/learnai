# BUG: accepts with probability min(1, q[x] / p[x]) instead of min(1, p[x] / q[x]), favoring exactly the tokens the draft model overrates.
import torch


def speculative_step(
    p: torch.Tensor, q: torch.Tensor, x: int, u_accept: float, u_resample: float
) -> tuple[int, bool]:
    if bool(u_accept * p[x] < q[x]):
        return x, True
    cdf = torch.cumsum(torch.clamp(p - q, min=0.0), dim=0)
    token = int((cdf <= u_resample * cdf[-1]).sum())
    return token, False
