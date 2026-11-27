import mlx.core as mx


def length_mask(lengths: mx.array, T: int, side: str) -> mx.array:
    pos = mx.arange(T)[None, :]  # (1, T): the position j of every column
    n = lengths[:, None]  # (B, 1): the length of every row
    if side == "right":
        return pos < n  # the first n positions are real
    return pos >= T - n  # the last n positions are real


def band_mask(T: int, w: int) -> mx.array:
    i = mx.arange(T)[:, None]  # (T, 1): the row index
    j = mx.arange(T)[None, :]  # (1, T): the column index
    back = i - j  # (T, T): how far column j is behind row i
    return (back >= 0) & (back < w)


def same_segment(seg: mx.array) -> mx.array:
    # A (T, 1) column against a (1, T) row: every pair of positions.
    return seg[:, None] == seg[None, :]
