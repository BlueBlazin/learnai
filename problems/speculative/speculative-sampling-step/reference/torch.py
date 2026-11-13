import torch


def speculative_step(
    p: torch.Tensor, q: torch.Tensor, x: int, u_accept: float, u_resample: float
) -> tuple[int, bool]:
    # Accept with probability min(1, p[x] / q[x]). Multiplying through by q[x] avoids
    # the division and needs no min: when p[x] >= q[x] the test holds for every u < 1.
    if bool(u_accept * q[x] < p[x]):
        return x, True
    # Rejected: sample from the residual max(0, p - q), normalized. Scaling the uniform
    # by the total (the last cumulative sum) is the same as normalizing the residual.
    cdf = torch.cumsum(torch.clamp(p - q, min=0.0), dim=0)
    # Inverse CDF: the first index whose cumulative sum exceeds the scaled uniform,
    # which is the number of entries that do not exceed it.
    token = int((cdf <= u_resample * cdf[-1]).sum())
    return token, False
