import mlx.core as mx


def fill_masked(scores: mx.array, keep: mx.array) -> mx.array:
    # where selects: the entry that is not kept is never used in any arithmetic, so it
    # does not matter what it holds.
    return mx.where(keep, scores, float("-inf"))


def masked_mean(x: mx.array, mask: mx.array) -> mx.array:
    # (B, T) -> (B, T, 1), so the mask lines up with the steps and broadcasts over the
    # features. A dropped step becomes 0, which adds nothing to the sum.
    kept = mx.where(mask[:, :, None], x, 0.0)
    count = mask.sum(axis=1)[:, None]  # (B, 1): kept steps per row
    # A row that keeps nothing has sum 0 and count 0; dividing by 1 returns its zeros.
    return kept.sum(axis=1) / mx.maximum(count, 1)


def masked_max(x: mx.array, mask: mx.array) -> mx.array:
    # A dropped step becomes -inf, which no kept value loses to. A row that keeps
    # nothing is all -inf, and so is its max.
    return mx.where(mask[:, :, None], x, float("-inf")).max(axis=1)
