import mlx.core as mx


def exclusive_cumsum(x: mx.array) -> mx.array:
    # MLX has the exclusive form built in: with inclusive=False, position i holds the
    # sum of the entries before it and position 0 holds 0.
    return mx.cumsum(x, axis=-1, inclusive=False)


def first_true(mask: mx.array) -> mx.array:
    # The running count of true entries is 0 exactly at the positions before the first
    # one. There are as many of those as the index of the first true entry, and T of
    # them when the row has none.
    seen = mx.cumsum(mask, axis=1)  # (B, T) int32
    return (seen == 0).sum(axis=1)


def through_first_eos(ids: mx.array, eos: int) -> mx.array:
    # Position t is kept when no eos comes strictly before it. The exclusive count
    # leaves the first eos itself at 0, so it is kept, and everything after it is not.
    is_eos = (ids == eos).astype(mx.int32)
    return exclusive_cumsum(is_eos) == 0
