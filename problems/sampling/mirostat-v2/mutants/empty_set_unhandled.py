# BUG: does not keep the most likely token when every surprise exceeds mu; the kept mass is 0 and the count of cumulative sums at or below 0 is clamped to the last token.
import math

import torch


def mirostat_step(
    logits: torch.Tensor, mu: torch.Tensor, tau: float, eta: float, u: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    log_p = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    surprise = -log_p / math.log(2.0)
    keep = surprise <= mu[:, None]
    cdf = torch.cumsum(torch.where(keep, torch.exp(log_p), 0.0), dim=-1)
    token = (cdf <= u[:, None] * cdf[:, -1:]).sum(dim=-1).clamp(max=logits.shape[-1] - 1)
    observed = torch.take_along_dim(surprise, token[:, None], dim=-1)[:, 0]
    return token, mu - eta * (observed - tau)
