import torch


def kl_from_logits(p_logits: torch.Tensor, q_logits: torch.Tensor) -> torch.Tensor:
    # Log-probabilities straight from the logits: finite wherever the logit is finite,
    # even when the probability itself underflows.
    log_p = torch.log_softmax(p_logits, dim=-1)
    log_q = torch.log_softmax(q_logits, dim=-1)
    # Zero probabilities are decided by the logits, never by an underflowed exp.
    p_zero = torch.isneginf(p_logits)
    q_zero = torch.isneginf(q_logits)
    terms = torch.exp(log_p) * (log_p - log_q)
    terms = torch.where(q_zero, torch.inf, terms)  # q = 0 < p: infinite, however small p is
    terms = torch.where(p_zero, 0.0, terms)  # p = 0: nothing, even where q = 0 too
    # The exact sum is >= 0; rounding can leave a tiny negative number.
    return terms.sum(dim=-1).clamp(min=0.0)
