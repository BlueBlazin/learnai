import mlx.core as mx


def token_scores(logits: mx.array, ids: mx.array) -> mx.array:
    # take_along_axis wants an index with as many axes as logits: (B, T) -> (B, T, 1).
    # The result has the index's shape, and the size-1 axis that was added is then
    # removed by name.
    return mx.take_along_axis(logits, ids[:, :, None], axis=2).squeeze(2)


def take_columns(values: mx.array, idx: mx.array) -> mx.array:
    # The index already has the rank of values: out[b, k] = values[b, idx[b, k]].
    return mx.take_along_axis(values, idx, axis=1)


def reorder_steps(x: mx.array, perm: mx.array) -> mx.array:
    # Along axis 1 this reads out[b, t, d] = x[b, index[b, t, d], d]. The (B, T, 1)
    # index is broadcast along the feature axis, so every feature of a step moves
    # together.
    return mx.take_along_axis(x, perm[:, :, None], axis=1)
