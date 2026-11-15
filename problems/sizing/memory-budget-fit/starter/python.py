def max_sequences(
    memory_bytes: int, weight_bytes: int, cfg: dict, dtype: str, context: int, overhead_percent: int
) -> int:
    """How many sequences of `context` tokens fit on the device at once.

    memory_bytes: the device's memory, in bytes.
    weight_bytes: the bytes the model's weights occupy.
    cfg: a dict with (at least) the int keys n_layers, n_kv_heads, head_dim, and
        block_tokens (the KV cache is allocated in blocks of that many tokens).
    dtype: the element type of the cache: "float32" (4 bytes), "float16" (2),
        "bfloat16" (2), or "int8" (1).
    context: tokens each sequence holds, >= 1.
    overhead_percent: the share of memory_bytes kept free of weights and cache, an
        int from 0 to 99.
    Returns an int >= 0.
    """
    raise NotImplementedError


def max_context(
    memory_bytes: int, weight_bytes: int, cfg: dict, dtype: str, batch: int, overhead_percent: int
) -> int:
    """The longest context, in tokens, that `batch` sequences can all hold at once.

    batch: number of sequences, >= 1. The other arguments are as in max_sequences.
    Returns an int >= 0, a multiple of cfg["block_tokens"].
    """
    raise NotImplementedError
