def count_params(cfg: dict) -> dict[str, int]:
    d = cfg["d_model"]
    q_width = cfg["n_heads"] * cfg["head_dim"]  # not always d_model (Qwen3-0.6B: 2048 vs 1024)
    kv_width = cfg["n_kv_heads"] * cfg["head_dim"]  # GQA: K and V are narrower than Q

    # W_q, W_k, W_v map d_model to the head widths; W_o maps the query width back.
    attention = d * q_width + 2 * d * kv_width + q_width * d
    if cfg["attn_bias"]:
        attention += q_width + 2 * kv_width + d  # one bias per output feature

    d_ff = cfg["d_ff"]
    if cfg["mlp_type"] == "swiglu":
        mlp = 3 * d * d_ff  # gate, up, down
        mlp_bias = 2 * d_ff + d
    else:
        mlp = 2 * d * d_ff  # up, down
        mlp_bias = d_ff + d
    if cfg["mlp_bias"]:
        mlp += mlp_bias

    per_norm = 2 * d if cfg["norm_bias"] else d
    n_layers = cfg["n_layers"]
    counts = {
        "embedding": cfg["vocab_size"] * d,
        "attention": n_layers * attention,
        "mlp": n_layers * mlp,
        "norm": (2 * n_layers + 1) * per_norm,  # two per block, one before the head
        "head": 0 if cfg["tie_embeddings"] else cfg["vocab_size"] * d,  # tied: same tensor
    }
    counts["total"] = sum(counts.values())
    return counts
