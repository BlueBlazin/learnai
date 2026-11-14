import mlx.core as mx


def split_heads(x: mx.array, n_heads: int) -> mx.array:
    batch, seq, width = x.shape
    # Cut the feature axis into (head, feature within head). This reshape only splits
    # the last axis, so no value moves: (B, T, D) -> (B, T, H, D / H).
    x = x.reshape(batch, seq, n_heads, width // n_heads)
    # Now move the head axis in front of the sequence axis: (B, H, T, D / H).
    return mx.swapaxes(x, 1, 2)


def merge_heads(x: mx.array) -> mx.array:
    batch, n_heads, seq, head_width = x.shape
    # Undo the two steps in reverse order: heads back next to the features, then glue
    # the two last axes together.
    return mx.swapaxes(x, 1, 2).reshape(batch, seq, n_heads * head_width)
