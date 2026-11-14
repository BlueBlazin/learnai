import mlx.core as mx


def verify_greedy(draft_tokens: mx.array, target_logits: mx.array) -> tuple[int, int]:
    """Verify K greedy draft tokens against the target model's logits.

    draft_tokens: integer array of shape (K,), the draft model's tokens.
    target_logits: float32 array of shape (K + 1, V). Row i scores the position of
        draft_tokens[i]; row K scores the position after the last draft token.
    Returns (n_accepted, next_token) as Python ints: the length of the longest draft
    prefix that equals the target's argmax, and the target's token for the position
    right after it (the correction at the first mismatch, or the bonus token).
    """
    raise NotImplementedError
