import math

import mlx.core as mx


def gqa(q: mx.array, k: mx.array, v: mx.array, mask: mx.array) -> mx.array:
    """Scaled dot-product attention in which groups of query heads share one key/value head.

    q: float32, shape (B, H, T, d). Queries, one set per query head.
    k, v: float32, shape (B, H_kv, S, d). Keys and values; H is a multiple of H_kv.
    mask: bool, shape (T, S). True where query position t may attend to key position s.
    Query head h uses key/value head h // (H // H_kv).
    Returns float32, shape (B, H, T, d).
    """
    raise NotImplementedError
