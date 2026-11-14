import torch


def sequence_logprob(
    logits: torch.Tensor, tokens: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Per-row log-probability of the tokens under next-token logits.

    logits: float32 tensor, shape (B, T, V). logits[b, t] scores the token at position t + 1.
    tokens: integer tensor, shape (B, T), values in [0, V).
    mask: bool tensor, shape (B, T). True marks a real token, False padding.
    Returns float32, shape (B,): the sum over t of
    log_softmax(logits[b, t])[tokens[b, t + 1]], over the pairs (t, t + 1) whose two
    positions are both real.
    """
    raise NotImplementedError


def perplexity(logits: torch.Tensor, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Per-row perplexity: exp(-sequence_logprob / number of counted pairs).

    Arguments as for sequence_logprob. Returns float32, shape (B,).
    """
    raise NotImplementedError
