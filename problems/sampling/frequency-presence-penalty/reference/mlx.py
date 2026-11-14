import mlx.core as mx


def count_generated(tokens: mx.array, prompt_lengths: mx.array, vocab_size: int) -> mx.array:
    # generated[b, t]: is position t of row b past that row's prompt?
    generated = mx.arange(tokens.shape[1]) >= prompt_lengths[:, None]
    # A histogram per row: add 1 at column tokens[b, t] for every generated position and
    # 0 for every prompt position. `.at[...].add` accumulates repeated indices, which is
    # the point: a token generated three times must count three.
    rows = mx.arange(tokens.shape[0])[:, None]
    counts = mx.zeros((tokens.shape[0], vocab_size), dtype=mx.int32)
    return counts.at[rows, tokens].add(generated.astype(mx.int32))


def apply_penalties(
    logits: mx.array,
    token_counts: mx.array,
    frequency_penalty: float,
    presence_penalty: float,
) -> mx.array:
    # Frequency grows with every occurrence; presence is charged once, as soon as the
    # token has occurred at all. Both are subtracted, whatever the sign of the logit.
    penalty = token_counts * frequency_penalty + (token_counts > 0) * presence_penalty
    return logits - penalty
