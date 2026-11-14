# BUG: decode_flops charges the dense context x context attention of a prefill, as if the KV cache did not exist and every earlier query were recomputed.
def _per_position_flops(cfg: dict) -> int:
    d = cfg["d_model"]
    q_width = cfg["n_heads"] * cfg["head_dim"]
    kv_width = cfg["n_kv_heads"] * cfg["head_dim"]
    projections = 2 * d * q_width + 2 * (2 * d * kv_width) + 2 * q_width * d
    n_mlp_matrices = 3 if cfg["mlp_type"] == "swiglu" else 2
    return projections + n_mlp_matrices * 2 * d * cfg["d_ff"]


def _attention_flops(cfg: dict, n_queries: int, n_keys: int) -> int:
    scores = 2 * n_queries * cfg["head_dim"] * n_keys
    mixing = 2 * n_queries * n_keys * cfg["head_dim"]
    return cfg["n_heads"] * (scores + mixing)


def _head_flops(cfg: dict) -> int:
    return 2 * cfg["d_model"] * cfg["vocab_size"]


def prefill_flops(cfg: dict, t: int) -> int:
    per_layer = t * _per_position_flops(cfg) + _attention_flops(cfg, t, t)
    return cfg["n_layers"] * per_layer + _head_flops(cfg)


def decode_flops(cfg: dict, context: int) -> int:
    per_layer = _per_position_flops(cfg) + _attention_flops(cfg, context, context)
    return cfg["n_layers"] * per_layer + _head_flops(cfg)
