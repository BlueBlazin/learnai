def _per_position_flops(cfg: dict) -> int:
    """FLOPs of one block's weight matmuls for one position (a matmul costs 2 m n p)."""
    d = cfg["d_model"]
    q_width = cfg["n_heads"] * cfg["head_dim"]
    kv_width = cfg["n_kv_heads"] * cfg["head_dim"]
    projections = 2 * d * q_width + 2 * (2 * d * kv_width) + 2 * q_width * d  # q, k, v, o
    n_mlp_matrices = 3 if cfg["mlp_type"] == "swiglu" else 2
    return projections + n_mlp_matrices * 2 * d * cfg["d_ff"]


def _attention_flops(cfg: dict, n_queries: int, n_keys: int) -> int:
    """Scores (Q K^T) and mixing (weights V) for every query head of one layer."""
    scores = 2 * n_queries * cfg["head_dim"] * n_keys
    mixing = 2 * n_queries * n_keys * cfg["head_dim"]
    return cfg["n_heads"] * (scores + mixing)  # GQA shares K and V, not the work


def _head_flops(cfg: dict) -> int:
    return 2 * cfg["d_model"] * cfg["vocab_size"]


def prefill_flops(cfg: dict, t: int) -> int:
    # t positions through every block, a dense t x t attention, and one row of logits.
    per_layer = t * _per_position_flops(cfg) + _attention_flops(cfg, t, t)
    return cfg["n_layers"] * per_layer + _head_flops(cfg)


def decode_flops(cfg: dict, context: int) -> int:
    # One position through every block; its single query attends over `context` keys.
    per_layer = _per_position_flops(cfg) + _attention_flops(cfg, 1, context)
    return cfg["n_layers"] * per_layer + _head_flops(cfg)
