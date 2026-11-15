# BUG: tokens masked in q are skipped (their term is set to 0) instead of making the divergence infinite.
import torch


def kl_from_logits(p_logits: torch.Tensor, q_logits: torch.Tensor) -> torch.Tensor:
    log_p = torch.log_softmax(p_logits, dim=-1)
    log_q = torch.log_softmax(q_logits, dim=-1)
    skip = torch.isneginf(p_logits) | torch.isneginf(q_logits)
    terms = torch.where(skip, 0.0, torch.exp(log_p) * (log_p - log_q))
    return terms.sum(dim=-1).clamp(min=0.0)
