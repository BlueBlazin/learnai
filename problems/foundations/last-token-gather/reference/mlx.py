import mlx.core as mx


def last_token_logits(logits: mx.array, lengths: mx.array, padding_side: str) -> mx.array:
    batch, seq, _ = logits.shape
    # The position of the last real token in each row.
    if padding_side == "right":
        last = lengths - 1  # real tokens are 0 .. length - 1
    elif padding_side == "left":
        last = mx.full(lengths.shape, seq - 1, dtype=lengths.dtype)  # the last column
    else:
        raise ValueError(f"padding_side must be 'left' or 'right', got {padding_side!r}")
    # Pair row b with position last[b]: two index arrays of shape (B,) pick one
    # (V,)-vector per row. A slice (logits[:, last]) would pick every position for
    # every row instead.
    rows = mx.arange(batch)
    return logits[rows, last]
