# BUG: walks the kept tokens from most to least likely (the candidate order of llama.cpp) instead of in token order, so the same u selects a different token.
import math

import torch


def mirostat_step(
    logits: torch.Tensor, mu: torch.Tensor, tau: float, eta: float, u: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    log_p = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    surprise = -log_p / math.log(2.0)
    top = torch.argmax(logits, dim=-1, keepdim=True)
    index = torch.arange(logits.shape[-1])
    keep = (surprise <= mu[:, None]) | (index == top)
    probs, order = torch.sort(
        torch.where(keep, torch.exp(log_p), 0.0), dim=-1, descending=True, stable=True
    )
    cdf = torch.cumsum(probs, dim=-1)
    rank = (cdf <= u[:, None] * cdf[:, -1:]).sum(dim=-1)
    token = torch.take_along_dim(order, rank[:, None], dim=-1)[:, 0]
    observed = torch.take_along_dim(surprise, token[:, None], dim=-1)[:, 0]
    return token, mu - eta * (observed - tau)
