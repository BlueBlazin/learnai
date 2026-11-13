import mlx.core as mx


def apply_repetition_penalty(
    logits: mx.array, prev_tokens: mx.array, penalty: float, pad_id: int
) -> mx.array:
    # seen[b, v]: does token v occur in row b's history? One scatter along the vocab
    # axis marks every history entry of every row. A token that occurs several times is
    # written several times with the same value, so it is still penalized once, and the
    # value written for a padding entry is False, so pad_id is never marked.
    seen = mx.put_along_axis(
        mx.zeros(logits.shape, dtype=mx.bool_), prev_tokens, prev_tokens != pad_id, axis=1
    )
    # Push the logit toward "less likely" whatever its sign.
    penalized = mx.where(logits > 0, logits / penalty, logits * penalty)
    return mx.where(seen, penalized, logits)
