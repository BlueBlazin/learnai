# BUG: sizes the cache with n_heads (the query heads) instead of n_kv_heads, ignoring that grouped-query attention stores only the shared key/value heads.
ITEMSIZE = {"float32": 4, "float16": 2, "bfloat16": 2, "int8": 1}


def kv_bytes_per_token(cfg: dict, dtype: str) -> int:
    return 2 * cfg["n_layers"] * cfg["n_heads"] * cfg["head_dim"] * ITEMSIZE[dtype]


def kv_bytes(cfg: dict, dtype: str, batch: int, context: int, window_layers: dict[int, int]) -> int:
    layer_tokens = sum(
        min(context, window_layers.get(layer, context)) for layer in range(cfg["n_layers"])
    )
    per_layer_token = 2 * cfg["n_heads"] * cfg["head_dim"] * ITEMSIZE[dtype]
    return batch * layer_tokens * per_layer_token
