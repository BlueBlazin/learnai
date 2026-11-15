import math

import mlx.core as mx


def attention_with_sinks(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    mask: mx.array,
    sinks: mx.array,
) -> mx.array:
    """Multi-head attention whose softmax has one extra, learned logit per head.

    q: float32, shape (B, H, T, d). Queries.
    k, v: float32, shape (B, H, S, d). Keys and values.
    mask: bool, shape (T, S). True where query position t may attend to key position s.
        A row may be all False.
    sinks: float32, shape (H,). The sink logit of each head; an entry may be -inf.
    Returns float32, shape (B, H, T, d): the values weighted by a softmax taken over
    the S scores q.k / sqrt(d) and the head's sink logit; the sink's share of the
    probability is discarded.
    """
    raise NotImplementedError
