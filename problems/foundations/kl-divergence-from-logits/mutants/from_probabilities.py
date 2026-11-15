# BUG: works on probabilities: log(softmax(q)) is -inf for a token whose probability underflows, so the row becomes +inf although its exact divergence is finite.
import torch


def kl_from_logits(p_logits: torch.Tensor, q_logits: torch.Tensor) -> torch.Tensor:
    p = torch.softmax(p_logits, dim=-1)
    q = torch.softmax(q_logits, dim=-1)
    terms = p * (torch.log(p) - torch.log(q))
    terms = torch.where(torch.isneginf(q_logits), torch.inf, terms)
    terms = torch.where(p == 0, 0.0, terms)
    return terms.sum(dim=-1).clamp(min=0.0)
