# BUG: computes log(softmax(logits)); a probability below about 1e-45 underflows to 0 and its log is -inf although the log-probability is representable.
import torch


def _pair_logprobs(
    logits: torch.Tensor, tokens: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Log-probability of each counted (context t, target t + 1) pair, 0 elsewhere."""
    logp = torch.log(torch.softmax(logits[:, :-1], dim=-1))
    targets = tokens[:, 1:]
    picked = torch.gather(logp, 2, targets[:, :, None])[:, :, 0]
    counted = mask[:, :-1] & mask[:, 1:]
    return torch.where(counted, picked, torch.zeros_like(picked)), counted


def sequence_logprob(
    logits: torch.Tensor, tokens: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    picked, _ = _pair_logprobs(logits, tokens, mask)
    return picked.sum(dim=1)


def perplexity(logits: torch.Tensor, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    picked, counted = _pair_logprobs(logits, tokens, mask)
    return torch.exp(-picked.sum(dim=1) / counted.sum(dim=1))
