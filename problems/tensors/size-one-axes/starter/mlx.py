import mlx.core as mx


def normalize_over(x: mx.array, dim: int) -> mx.array:
    """Divide x by its sum over one axis.

    x: float32 with 1 to 4 axes; every entry is positive.
    dim: the axis to sum over; negative values count from the end.
    Returns float32 of the same shape as x; every slice along dim sums to 1.
    """
    raise NotImplementedError


def last_step(x: mx.array) -> mx.array:
    """The last step of every row.

    x: float32, shape (B, T, D).
    Returns float32, shape (B, D), with out[b] = x[b, T - 1]. The shape is (B, D) also
    when B, T, or D is 1.
    """
    raise NotImplementedError


def column(x: mx.array, j: int) -> mx.array:
    """One column of a matrix, still as a column.

    x: float32, shape (B, V).
    j: a Python int, 0 <= j < V.
    Returns float32, shape (B, 1), with out[b, 0] = x[b, j].
    """
    raise NotImplementedError
