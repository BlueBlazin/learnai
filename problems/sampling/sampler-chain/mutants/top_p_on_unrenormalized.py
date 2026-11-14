# BUG: computes the nucleus from the probabilities of the full distribution instead of the distribution renormalized over the top-k survivors, so the survivors' mass reaches p later and too many tokens are kept.
import torch

from whetstone.kit import ops


def penalize(logits: torch.Tensor, cfg: dict, history: torch.Tensor) -> torch.Tensor:
    vocab = torch.arange(logits.shape[-1])
    counts = (history[:, :, None] == vocab).sum(dim=1)
    seen = counts > 0
    penalty = cfg["repetition_penalty"]
    x = torch.where(seen, torch.where(logits > 0, logits / penalty, logits * penalty), logits)
    return x - counts * cfg["frequency_penalty"] - seen * cfg["presence_penalty"]


def chain_logits(logits: torch.Tensor, cfg: dict, history: torch.Tensor) -> torch.Tensor:
    x = penalize(logits, cfg, history)
    kept = ops.top_k_filter(x, cfg["top_k"])
    if cfg["top_p"] < 1.0:
        # The nucleus of the unfiltered distribution, intersected with the top-k set.
        kept = torch.where(ops.top_p_filter(x, cfg["top_p"]) > -torch.inf, kept, -torch.inf)
    kept = ops.min_p_filter(kept, cfg["min_p"])
    return kept / cfg["temperature"]


def sample_chain(
    logits: torch.Tensor, cfg: dict, history: torch.Tensor, u: torch.Tensor
) -> torch.Tensor:
    probs = ops.softmax(chain_logits(logits, cfg, history))
    cdf = torch.cumsum(probs, dim=-1)
    return (cdf <= u[:, None] * cdf[:, -1:]).sum(dim=-1)
