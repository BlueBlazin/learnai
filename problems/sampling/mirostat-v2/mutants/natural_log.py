# BUG: measures surprise in nats (-ln p) instead of bits, so mu and tau are compared with numbers that are 0.69 times too small.
import torch


def mirostat_step(
    logits: torch.Tensor, mu: torch.Tensor, tau: float, eta: float, u: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    log_p = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    surprise = -log_p
    top = torch.argmax(logits, dim=-1, keepdim=True)
    index = torch.arange(logits.shape[-1])
    keep = (surprise <= mu[:, None]) | (index == top)
    cdf = torch.cumsum(torch.where(keep, torch.exp(log_p), 0.0), dim=-1)
    token = (cdf <= u[:, None] * cdf[:, -1:]).sum(dim=-1)
    observed = torch.take_along_dim(surprise, token[:, None], dim=-1)[:, 0]
    return token, mu - eta * (observed - tau)
