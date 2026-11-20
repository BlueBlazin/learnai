from typing import Any

import mlx.core as mx


def dsa_step(
    dsa: Any, layer: int, x_new: mx.array, cache: tuple[mx.array, Any], k: int
) -> tuple[mx.array, tuple[mx.array, Any]]:
    """One sparse-attention layer on T new tokens that continue S cached ones.

    dsa: a TinyDSA. For rows x of shape (n, d_model):
        dsa.index_key(layer, x)    -> (n, index_dim) indexer keys; counted
        dsa.index_query(layer, x)  -> (q_idx (n, index_heads, index_dim), w (n, index_heads))
        dsa.qkv(layer, x)          -> q, k, v, each (n, n_heads, head_dim)
        dsa.out(layer, attended)   -> (n, d_model) from (n, n_heads, head_dim)
    layer: int. Which layer of dsa.
    x_new: float32, shape (T, d_model). The inputs of the new tokens, oldest first.
    cache: (index_keys, store) for the S tokens before them: index_keys is float32 of
        shape (S, index_dim), and store is a CountingKV holding their keys and values
        (store.n == S; store.append(k, v) -> a longer store; store.gather(rows, valid)).
    k: int >= 1. How many tokens each query selects.
    Returns (out, cache): out is float32 of shape (T, d_model), and the cache now holds
    S + T tokens.
    """
    raise NotImplementedError
