import torch


def _pair_logprobs(
    logits: torch.Tensor, tokens: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Log-probability of each counted (context t, target t + 1) pair, 0 elsewhere."""
    # logits[:, t] predicts tokens[:, t + 1]: drop the last position's logits (they
    # predict a token that is not in the batch) and the first token (nothing predicts it).
    logp = torch.log_softmax(logits[:, :-1], dim=-1)  # (B, T - 1, V)
    targets = tokens[:, 1:]  # (B, T - 1)
    picked = torch.gather(logp, 2, targets[:, :, None])[:, :, 0]  # (B, T - 1)
    # A pair counts when both of its positions are real tokens.
    counted = mask[:, :-1] & mask[:, 1:]
    # where, not multiplication: an uncounted entry may be -inf, and 0 * -inf is NaN.
    return torch.where(counted, picked, torch.zeros_like(picked)), counted


def sequence_logprob(
    logits: torch.Tensor, tokens: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    picked, _ = _pair_logprobs(logits, tokens, mask)
    return picked.sum(dim=1)


def perplexity(logits: torch.Tensor, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    picked, counted = _pair_logprobs(logits, tokens, mask)
    # exp of the mean negative log-probability per counted pair, not of the sum.
    return torch.exp(-picked.sum(dim=1) / counted.sum(dim=1))
