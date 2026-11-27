import mlx.core as mx


def token_scores(logits: mx.array, ids: mx.array) -> mx.array:
    """The score of one chosen token at every position.

    logits: float32, shape (B, T, V).
    ids: int32, shape (B, T), with 0 <= ids[b, t] < V.
    Returns float32, shape (B, T), with out[b, t] = logits[b, t, ids[b, t]].
    """
    raise NotImplementedError


def take_columns(values: mx.array, idx: mx.array) -> mx.array:
    """K entries from every row, each row with its own choice.

    values: float32, shape (B, V).
    idx: int32, shape (B, K), with 0 <= idx[b, k] < V.
    Returns float32, shape (B, K), with out[b, k] = values[b, idx[b, k]].
    """
    raise NotImplementedError


def reorder_steps(x: mx.array, perm: mx.array) -> mx.array:
    """Reorder the steps of every row by that row's own permutation.

    x: float32, shape (B, T, D).
    perm: int32, shape (B, T); every row is a permutation of 0 .. T - 1.
    Returns float32, shape (B, T, D), with out[b, t] = x[b, perm[b, t]].
    """
    raise NotImplementedError
