import mlx.core as mx


def schedule(n_masked: int, steps: int) -> list[int]:
    # Spread the masked positions evenly; the first n_masked % steps steps take one
    # more, so the counts add up to n_masked and nothing is left for after the end.
    base, extra = divmod(n_masked, steps)
    return [base + 1 if i < extra else base for i in range(steps)]


def unmask_step(
    tokens: mx.array,
    logits: mx.array,
    mask_id: int,
    n_unmask: int,
    temperature: float,
    uniforms: mx.array,
) -> mx.array:
    masked = tokens == mask_id
    # The model never predicts the mask token itself.
    logits = mx.where(mx.arange(logits.shape[-1]) == mask_id, float("-inf"), logits)

    # A candidate token for every position, from that position's own uniform number.
    if temperature == 0:
        candidates = mx.argmax(logits, axis=-1)
    else:
        cdf = mx.cumsum(mx.softmax(logits / temperature, axis=-1), axis=-1)
        # Inverse CDF: the first index whose cumulative sum exceeds u times the total.
        candidates = (cdf <= uniforms[:, None] * cdf[:, -1:]).sum(axis=-1)

    # Confidence: the probability the model itself (temperature 1) gives the candidate
    # that was drawn, not the probability of its favorite token.
    probs = mx.softmax(logits, axis=-1)
    confidence = mx.take_along_axis(probs, candidates[:, None], axis=-1)[:, 0]
    # A position that is already unmasked can never be chosen.
    confidence = mx.where(masked, confidence, float("-inf"))

    # Commit the n most confident masked positions. mx.argsort is stable, so sorting
    # the negated values puts equal confidences in index order; rank[i] is the place
    # of position i in that order.
    n = min(n_unmask, int(masked.sum().item()))
    order = mx.argsort(-confidence)
    rank = mx.argsort(order)
    return mx.where(rank < n, candidates.astype(tokens.dtype), tokens)
