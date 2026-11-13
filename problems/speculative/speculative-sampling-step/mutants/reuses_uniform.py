# BUG: uses u_accept for the resampling draw as well; given a rejection that number is no longer uniform on [0, 1), so the resampled token is biased.
import torch


def speculative_step(
    p: torch.Tensor, q: torch.Tensor, x: int, u_accept: float, u_resample: float
) -> tuple[int, bool]:
    if bool(u_accept * q[x] < p[x]):
        return x, True
    cdf = torch.cumsum(torch.clamp(p - q, min=0.0), dim=0)
    token = int((cdf <= u_accept * cdf[-1]).sum())
    return token, False
