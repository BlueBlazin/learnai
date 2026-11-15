# BUG: no special case for zero probabilities: a token masked in p gives 0 * -inf (or 0 * (-inf - -inf)), which is NaN instead of 0.
import torch


def kl_from_logits(p_logits: torch.Tensor, q_logits: torch.Tensor) -> torch.Tensor:
    log_p = torch.log_softmax(p_logits, dim=-1)
    log_q = torch.log_softmax(q_logits, dim=-1)
    terms = torch.exp(log_p) * (log_p - log_q)
    return terms.sum(dim=-1).clamp(min=0.0)
