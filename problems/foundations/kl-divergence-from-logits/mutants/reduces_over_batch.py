# BUG: averages over the rows as well (the 'batchmean' habit) and returns one number instead of one value per row.
import torch


def kl_from_logits(p_logits: torch.Tensor, q_logits: torch.Tensor) -> torch.Tensor:
    log_p = torch.log_softmax(p_logits, dim=-1)
    log_q = torch.log_softmax(q_logits, dim=-1)
    p_zero = torch.isneginf(p_logits)
    q_zero = torch.isneginf(q_logits)
    terms = torch.exp(log_p) * (log_p - log_q)
    terms = torch.where(q_zero, torch.inf, terms)
    terms = torch.where(p_zero, 0.0, terms)
    return terms.sum(dim=-1).clamp(min=0.0).mean()
