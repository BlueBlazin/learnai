# BUG: decides p = 0 from the underflowed probability exp(log_p) instead of from the logit, so a token with a finite p logit that q masks contributes 0 instead of +inf.
import torch


def kl_from_logits(p_logits: torch.Tensor, q_logits: torch.Tensor) -> torch.Tensor:
    log_p = torch.log_softmax(p_logits, dim=-1)
    log_q = torch.log_softmax(q_logits, dim=-1)
    p = torch.exp(log_p)
    terms = torch.where(p == 0, 0.0, p * (log_p - log_q))
    return terms.sum(dim=-1).clamp(min=0.0)
