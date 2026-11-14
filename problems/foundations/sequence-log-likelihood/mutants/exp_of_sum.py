# BUG: perplexity is exp of the negative sum instead of the negative mean, so it grows exponentially with the sequence length.
import torch


def _pair_logprobs(
    logits: torch.Tensor, tokens: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Log-probability of each counted (context t, target t + 1) pair, 0 elsewhere."""
    logp = torch.log_softmax(logits[:, :-1], dim=-1)
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
    picked, _ = _pair_logprobs(logits, tokens, mask)
    return torch.exp(-picked.sum(dim=1))
