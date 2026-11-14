import mlx.core as mx


def batch_generate(
    model, prompts: list[list[int]], max_new_tokens: int, eos_id: int | None
) -> list[list[int]]:
    """Greedy generation for prompts of different lengths, in one left-padded batch.

    model: an MLXTinyLM, called with a (B, T) integer array:
        logits, cache = model(tokens, mask=mask, positions=positions)
        logits, cache = model(tokens, cache, mask=mask, positions=positions)
        mask: additive float array broadcastable to (B, 1, T, cached + T); it replaces
            the causal mask. positions: integer array (B, T), the RoPE positions.
        logits: (B, T, vocab_size); everything returned is lazy.
    prompts: B >= 1 lists of token ids, each with at least one token.
    max_new_tokens: >= 1, the limit for every row.
    eos_id: a row stops after generating this token (it is kept); None: never.
    Returns B lists of Python ints: for each prompt, exactly the tokens that greedy
    decoding of that prompt alone generates.
    """
    raise NotImplementedError
