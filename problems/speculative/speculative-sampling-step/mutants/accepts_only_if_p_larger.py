# BUG: accepts only when p[x] >= q[x] and otherwise always rejects, ignoring u_accept; tokens with p[x] < q[x] should still be accepted with probability p[x] / q[x].
import torch


def speculative_step(
    p: torch.Tensor, q: torch.Tensor, x: int, u_accept: float, u_resample: float
) -> tuple[int, bool]:
    if bool(p[x] >= q[x]):
        return x, True
    cdf = torch.cumsum(torch.clamp(p - q, min=0.0), dim=0)
    token = int((cdf <= u_resample * cdf[-1]).sum())
    return token, False
