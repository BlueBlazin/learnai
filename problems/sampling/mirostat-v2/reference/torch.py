import math

import torch


def mirostat_step(
    logits: torch.Tensor, mu: torch.Tensor, tau: float, eta: float, u: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    # Surprise of every token in bits, under the model's own distribution. A masked
    # token has log-probability -inf and surprise +inf.
    log_p = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    surprise = -log_p / math.log(2.0)
    # Keep the tokens that are no more surprising than mu. The most likely token is
    # the least surprising one, so "keep it always" is the same as "keep it when
    # nothing else passes".
    top = torch.argmax(logits, dim=-1, keepdim=True)
    index = torch.arange(logits.shape[-1])
    keep = (surprise <= mu[:, None]) | (index == top)
    # Inverse-CDF sampling over the kept tokens, in token order. Scaling u by the kept
    # mass is the renormalization.
    cdf = torch.cumsum(torch.where(keep, torch.exp(log_p), 0.0), dim=-1)
    token = (cdf <= u[:, None] * cdf[:, -1:]).sum(dim=-1)
    # Feedback: the surprise that was observed, before truncation, against the target.
    observed = torch.take_along_dim(surprise, token[:, None], dim=-1)[:, 0]
    return token, mu - eta * (observed - tau)
