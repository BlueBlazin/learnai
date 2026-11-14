# BUG: the element size is taken in bits (the number in the dtype name) instead of bytes, so every result is 8 times too large.
ITEMSIZE = {"float32": 32, "float16": 16, "bfloat16": 16, "int8": 8}


def kv_bytes_per_token(cfg: dict, dtype: str) -> int:
    return 2 * cfg["n_layers"] * cfg["n_kv_heads"] * cfg["head_dim"] * ITEMSIZE[dtype]


def kv_bytes(cfg: dict, dtype: str, batch: int, context: int, window_layers: dict[int, int]) -> int:
    layer_tokens = sum(
        min(context, window_layers.get(layer, context)) for layer in range(cfg["n_layers"])
    )
    per_layer_token = 2 * cfg["n_kv_heads"] * cfg["head_dim"] * ITEMSIZE[dtype]
    return batch * layer_tokens * per_layer_token
