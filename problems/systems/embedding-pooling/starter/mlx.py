import mlx.core as mx


def pool(states: mx.array, mask: mx.array, mode: str) -> mx.array:
    """One unit-length embedding per row, pooled from the states of its real tokens.

    states: float32, shape (B, T, D). The encoder's state at every position.
    mask: integer, shape (B, T). 1 at real tokens, 0 at padding; the real tokens of a
        row are contiguous, at the start (right padding) or at the end (left padding).
    mode: "mean" (average of the real tokens), "cls" (the first real token), or
        "last" (the last real token).
    Returns float32, shape (B, D): the pooled vectors divided by max(norm, 1e-12).
    """
    raise NotImplementedError


def truncate_matryoshka(e: mx.array, dim: int) -> mx.array:
    """Keep the first `dim` coordinates of each embedding and renormalize.

    e: float32, shape (..., D). dim: an int, 1 <= dim <= D.
    Returns float32, shape (..., dim): e[..., :dim] divided by max(norm, 1e-12).
    """
    raise NotImplementedError
