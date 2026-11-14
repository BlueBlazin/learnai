import math

import mlx.core as mx


def top_k_filter(logits: mx.array, k: int) -> mx.array:
    """Keep the k largest logits of every row and set the others to -inf.

    logits: float32 array of shape (B, V); entries may be -inf, finite entries of a
        row are distinct. It may be lazy and must stay lazy.
    k: int >= 1; with k >= V the logits are returned unchanged.
    Returns a float32 array of shape (B, V), built without evaluating anything and
    without modifying `logits`.
    """
    raise NotImplementedError


def scatter_penalty(logits: mx.array, ids: mx.array, value: float) -> mx.array:
    """Subtract `value` once from the logit of every token that occurs in `ids`.

    logits: float32 array of shape (B, V).
    ids: integer array of shape (B, N), N >= 1, token ids in [0, V); a row may repeat
        an id, and the token is still penalized once.
    value: Python float.
    Returns a float32 array of shape (B, V), built without evaluating anything and
    without modifying `logits`.
    """
    raise NotImplementedError
