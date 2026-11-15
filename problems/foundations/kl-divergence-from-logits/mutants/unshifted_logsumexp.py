# BUG: log-probabilities are computed as x - logsumexp(x) without shifting by the row maximum first; at logits near 1e4 the subtraction loses everything below about 1e-3.
import torch


def kl_from_logits(p_logits: torch.Tensor, q_logits: torch.Tensor) -> torch.Tensor:
    log_p = p_logits - torch.logsumexp(p_logits, dim=-1, keepdim=True)
    log_q = q_logits - torch.logsumexp(q_logits, dim=-1, keepdim=True)
    p_zero = torch.isneginf(p_logits)
    q_zero = torch.isneginf(q_logits)
    terms = torch.exp(log_p) * (log_p - log_q)
    terms = torch.where(q_zero, torch.inf, terms)
    terms = torch.where(p_zero, 0.0, terms)
    return terms.sum(dim=-1).clamp(min=0.0)
