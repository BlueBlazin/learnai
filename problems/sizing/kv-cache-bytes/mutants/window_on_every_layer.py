# BUG: once any layer has a sliding window, the smallest window is applied to every layer, global layers included.
ITEMSIZE = {"float32": 4, "float16": 2, "bfloat16": 2, "int8": 1}


def kv_bytes_per_token(cfg: dict, dtype: str) -> int:
    return 2 * cfg["n_layers"] * cfg["n_kv_heads"] * cfg["head_dim"] * ITEMSIZE[dtype]


def kv_bytes(cfg: dict, dtype: str, batch: int, context: int, window_layers: dict[int, int]) -> int:
    window = min(window_layers.values(), default=context)
    layer_tokens = cfg["n_layers"] * min(context, window)
    per_layer_token = 2 * cfg["n_kv_heads"] * cfg["head_dim"] * ITEMSIZE[dtype]
    return batch * layer_tokens * per_layer_token
