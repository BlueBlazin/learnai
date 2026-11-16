# BUG: divides by the temperature before the filters (the transformers and vLLM order), so top-p and min-p measure the tempered distribution and keep a different set of tokens.
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
    x = penalize(logits, cfg, history) / cfg["temperature"]
    x = ops.top_k_filter(x, cfg["top_k"])
    if cfg["top_p"] < 1.0:
        x = ops.top_p_filter(x, cfg["top_p"])
    return ops.min_p_filter(x, cfg["min_p"])


def sample_chain(
    logits: torch.Tensor, cfg: dict, history: torch.Tensor, u: torch.Tensor
) -> torch.Tensor:
    probs = ops.softmax(chain_logits(logits, cfg, history))
    cdf = torch.cumsum(probs, dim=-1)
    return (cdf <= u[:, None] * cdf[:, -1:]).sum(dim=-1)
