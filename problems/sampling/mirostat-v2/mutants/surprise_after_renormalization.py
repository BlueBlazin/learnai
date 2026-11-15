# BUG: measures the observed surprise with the renormalized probability of the truncated distribution (as llama.cpp does), which is smaller than the token's surprise under the model.
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
    cdf = torch.cumsum(torch.where(keep, torch.exp(log_p), 0.0), dim=-1)
    token = (cdf <= u[:, None] * cdf[:, -1:]).sum(dim=-1)
    kept_bits = torch.log2(cdf[:, -1])
    observed = torch.take_along_dim(surprise, token[:, None], dim=-1)[:, 0] + kept_bits
    return token, mu - eta * (observed - tau)
