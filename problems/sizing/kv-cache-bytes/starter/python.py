def kv_bytes_per_token(cfg: dict, dtype: str) -> int:
    """Bytes of KV cache that one token of one sequence occupies, over all layers.

    cfg: a dict with (at least) the int keys n_layers, n_kv_heads, and head_dim.
    dtype: the element type of the cache: "float32" (4 bytes), "float16" (2),
        "bfloat16" (2), or "int8" (1).
    Returns an int: every layer stores one key and one value vector of
    n_kv_heads * head_dim elements for the token.
    """
    raise NotImplementedError


def kv_bytes(cfg: dict, dtype: str, batch: int, context: int, window_layers: dict[int, int]) -> int:
    """Bytes of KV cache for `batch` sequences that each hold `context` tokens.

    batch: number of sequences, >= 1.
    context: tokens per sequence, >= 0.
    window_layers: maps a layer index in range(n_layers) to that layer's sliding
        window, in tokens. Such a layer stores min(context, window) tokens; a layer
        that is not a key stores all `context` tokens. May be empty.
    Returns an int.
    """
    raise NotImplementedError
