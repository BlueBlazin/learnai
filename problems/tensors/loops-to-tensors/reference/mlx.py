import mlx.core as mx


def segment_means(x: mx.array, seg: mx.array, S: int) -> mx.array:
    # An adding scatter along axis 0: row n of x is added to row seg[n] of the sums,
    # and a 1 to entry seg[n] of the counts. Rows that share a segment accumulate.
    sums = mx.zeros((S, x.shape[1]), dtype=x.dtype).at[seg].add(x)
    counts = mx.zeros((S,), dtype=x.dtype).at[seg].add(1.0)
    # An empty segment has sum 0 and count 0; dividing by 1 there leaves the 0.
    return sums / mx.maximum(counts, 1.0)[:, None]


def common_prefix_len(a: mx.array, b: mx.array) -> mx.array:
    # The running product of the 0/1 matches is 1 while every position so far has
    # matched and 0 from the first mismatch on, whatever comes after it.
    still_equal = mx.cumprod((a == b).astype(mx.int32), axis=1)  # (B, T)
    return still_equal.sum(axis=1)


def repeated_bigram_ban(ids: mx.array, V: int) -> mx.array:
    # A pair starts at every position except the final one. Position t starts a pair
    # that bans its second token when its first token is the row's last token.
    first = ids[:, :-1]  # (B, T - 1)
    second = ids[:, 1:]  # (B, T - 1)
    starts_with_last = (first == ids[:, -1:]).astype(mx.int32)
    # Count, for every token v, the pairs (last, v): an adding scatter of 0s and 1s
    # into column second[b, t]. A setting scatter of False could erase a True.
    rows = mx.arange(ids.shape[0])[:, None]
    counts = mx.zeros((ids.shape[0], V), dtype=mx.int32).at[rows, second].add(starts_with_last)
    return counts > 0
