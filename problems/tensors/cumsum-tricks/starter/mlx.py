import mlx.core as mx


def exclusive_cumsum(x: mx.array) -> mx.array:
    """The sum of the entries before each position, along the last axis.

    x: float32, with 1 to 3 axes.
    Returns float32 of the same shape as x, with out[..., i] = x[..., 0] + ... +
    x[..., i - 1]. out[..., 0] is 0.
    """
    raise NotImplementedError


def first_true(mask: mx.array) -> mx.array:
    """The index of the first true entry of every row.

    mask: bool, shape (B, T).
    Returns an integer tensor, shape (B,): the smallest t with mask[b, t] true, and T
    for a row with no true entry.
    """
    raise NotImplementedError


def through_first_eos(ids: mx.array, eos: int) -> mx.array:
    """The mask of each row up to and including its first eos token.

    ids: int32, shape (B, T).
    eos: the id of the end-of-sequence token, a Python int.
    Returns bool, shape (B, T): true at the positions up to and including the first
    position where ids[b, t] == eos, false after it. A row without eos is all true.
    """
    raise NotImplementedError
