# BUG: assumes n_heads * head_dim equals d_model: the query width is taken to be d_model and the KV width d_model * n_kv_heads // n_heads, ignoring head_dim.
def count_params(cfg: dict) -> dict[str, int]:
    d = cfg["d_model"]
    q_width = d
    kv_width = d * cfg["n_kv_heads"] // cfg["n_heads"]

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
