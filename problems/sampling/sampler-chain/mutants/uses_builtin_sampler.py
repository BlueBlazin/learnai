# BUG: draws the token with torch.multinomial instead of the supplied uniform numbers, so the result is not a function of u.
import torch

from learnai.kit import ops


def penalize(logits: torch.Tensor, cfg: dict, history: torch.Tensor) -> torch.Tensor:
    vocab = torch.arange(logits.shape[-1])
    counts = (history[:, :, None] == vocab).sum(dim=1)
    seen = counts > 0
    penalty = cfg["repetition_penalty"]
    x = torch.where(seen, torch.where(logits > 0, logits / penalty, logits * penalty), logits)
    return x - counts * cfg["frequency_penalty"] - seen * cfg["presence_penalty"]


def chain_logits(logits: torch.Tensor, cfg: dict, history: torch.Tensor) -> torch.Tensor:
    x = penalize(logits, cfg, history)
    x = ops.top_k_filter(x, cfg["top_k"])
    if cfg["top_p"] < 1.0:
        x = ops.top_p_filter(x, cfg["top_p"])
    x = ops.min_p_filter(x, cfg["min_p"])
    return x / cfg["temperature"]


def sample_chain(
    logits: torch.Tensor, cfg: dict, history: torch.Tensor, u: torch.Tensor
) -> torch.Tensor:
    probs = ops.softmax(chain_logits(logits, cfg, history))
    return torch.multinomial(probs, 1)[:, 0]
