# BUG: updates mu correctly but samples from the full distribution: the limit is never applied.
import math

import torch


def mirostat_step(
    logits: torch.Tensor, mu: torch.Tensor, tau: float, eta: float, u: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    log_p = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    surprise = -log_p / math.log(2.0)
    cdf = torch.cumsum(torch.exp(log_p), dim=-1)
    token = (cdf <= u[:, None] * cdf[:, -1:]).sum(dim=-1)
    observed = torch.take_along_dim(surprise, token[:, None], dim=-1)[:, 0]
    return token, mu - eta * (observed - tau)
