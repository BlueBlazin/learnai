# BUG: the key and value projections are sized with n_heads like the query projection, ignoring that grouped-query attention has only n_kv_heads of them.
def count_params(cfg: dict) -> dict[str, int]:
    d = cfg["d_model"]
    q_width = cfg["n_heads"] * cfg["head_dim"]
    kv_width = cfg["n_heads"] * cfg["head_dim"]

    attention = d * q_width + 2 * d * kv_width + q_width * d
    if cfg["attn_bias"]:
        attention += q_width + 2 * kv_width + d

    d_ff = cfg["d_ff"]
    if cfg["mlp_type"] == "swiglu":
        mlp = 3 * d * d_ff
        mlp_bias = 2 * d_ff + d
    else:
        mlp = 2 * d * d_ff
        mlp_bias = d_ff + d
    if cfg["mlp_bias"]:
        mlp += mlp_bias

    per_norm = 2 * d if cfg["norm_bias"] else d
    n_layers = cfg["n_layers"]
    counts = {
        "embedding": cfg["vocab_size"] * d,
        "attention": n_layers * attention,
        "mlp": n_layers * mlp,
        "norm": (2 * n_layers + 1) * per_norm,
        "head": 0 if cfg["tie_embeddings"] else cfg["vocab_size"] * d,
    }
    counts["total"] = sum(counts.values())
    return counts
