import mlx.core as mx


def length_mask(lengths: mx.array, T: int, side: str) -> mx.array:
    """The mask of real tokens in a padded batch.

    lengths: int32, shape (B,), with 0 <= lengths[b] <= T: the real tokens of row b.
    T: the padded length, a positive Python int.
    side: "right" (padding after the real tokens) or "left" (padding before them).
    Returns bool, shape (B, T). Row b has lengths[b] true entries: the first ones for
    "right", the last ones for "left".
    """
    raise NotImplementedError


def band_mask(T: int, w: int) -> mx.array:
    """The causal window of width w.

    T: the number of positions, a positive Python int.
    w: the window width, a positive Python int; it may be larger than T.
    Returns bool, shape (T, T), with out[i, j] true when 0 <= i - j < w.
    """
    raise NotImplementedError


def same_segment(seg: mx.array) -> mx.array:
    """Which pairs of positions are in the same segment.

    seg: int32, shape (T,): a segment id per position, any non-negative integers.
    Returns bool, shape (T, T), with out[i, j] true when seg[i] == seg[j].
    """
    raise NotImplementedError
