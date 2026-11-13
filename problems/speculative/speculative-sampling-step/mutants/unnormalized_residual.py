# BUG: compares u_resample with the cumulative sum of the unnormalized residual, whose total is below 1, so large uniforms fall off the end and the last token is emitted.
import torch


def speculative_step(
    p: torch.Tensor, q: torch.Tensor, x: int, u_accept: float, u_resample: float
) -> tuple[int, bool]:
    if bool(u_accept * q[x] < p[x]):
        return x, True
    cdf = torch.cumsum(torch.clamp(p - q, min=0.0), dim=0)
    token = min(int((cdf <= u_resample).sum()), p.shape[0] - 1)
    return token, False
