import mlx.core as mx


def sequence_logprob(logits: mx.array, tokens: mx.array, mask: mx.array) -> mx.array:
    """Per-row log-probability of the tokens under next-token logits.

    logits: float32 array, shape (B, T, V). logits[b, t] scores the token at position t + 1.
    tokens: integer array, shape (B, T), values in [0, V).
    mask: bool array, shape (B, T). True marks a real token, False padding.
    Returns float32, shape (B,): the sum over t of
    log_softmax(logits[b, t])[tokens[b, t + 1]], over the pairs (t, t + 1) whose two
    positions are both real.
    """
    raise NotImplementedError


def perplexity(logits: mx.array, tokens: mx.array, mask: mx.array) -> mx.array:
    """Per-row perplexity: exp(-sequence_logprob / number of counted pairs).

    Arguments as for sequence_logprob. Returns float32, shape (B,).
    """
    raise NotImplementedError
