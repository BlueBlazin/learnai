import torch

from learnai.kit import ops


def chain_logits(logits: torch.Tensor, cfg: dict, history: torch.Tensor) -> torch.Tensor:
    # 1. Penalties, on the raw logits. counts[b, v] = how often token v occurs in row b
    #    of the history.
    vocab = torch.arange(logits.shape[-1])
    counts = (history[:, :, None] == vocab).sum(dim=1)
    seen = counts > 0
    penalty = cfg["repetition_penalty"]
    # Repetition penalty: always toward "less likely", so positive logits are divided
    # and the others multiplied.
    x = torch.where(seen, torch.where(logits > 0, logits / penalty, logits * penalty), logits)
    x = x - counts * cfg["frequency_penalty"] - seen * cfg["presence_penalty"]

    # 2-4. The truncation filters. Each one sees the softmax of what the previous one
    #      left, because removed entries are -inf.
    x = ops.top_k_filter(x, cfg["top_k"])
    if cfg["top_p"] < 1.0:  # 1.0 switches the stage off; do not let rounding trim the tail
        x = ops.top_p_filter(x, cfg["top_p"])
    x = ops.min_p_filter(x, cfg["min_p"])

    # 5. Temperature last: it reshapes the distribution over the survivors, but has no
    #    say in which tokens survive.
    return x / cfg["temperature"]


def sample_chain(
    logits: torch.Tensor, cfg: dict, history: torch.Tensor, u: torch.Tensor
) -> torch.Tensor:
    probs = ops.softmax(chain_logits(logits, cfg, history))
    # Inverse CDF: the first index whose cumulative sum exceeds u times the row total.
    cdf = torch.cumsum(probs, dim=-1)
    return (cdf <= u[:, None] * cdf[:, -1:]).sum(dim=-1)
