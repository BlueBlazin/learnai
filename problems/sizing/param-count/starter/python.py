def count_params(cfg: dict) -> dict[str, int]:
    """Count the parameters of a decoder-only transformer from its configuration.

    cfg: a dict with the keys
        vocab_size, d_model, n_layers, n_heads, n_kv_heads, head_dim, d_ff: ints
        mlp_type: "swiglu" (gate, up, down: three matrices) or "gelu" (up, down: two)
        tie_embeddings: bool, the output head reuses the embedding matrix
        attn_bias, mlp_bias: bool, the linear layers of that kind have bias vectors
        norm_bias: bool, each norm has a bias next to its gain (LayerNorm vs. RMSNorm)
    Returns a dict of ints with exactly the keys "embedding", "attention", "mlp",
    "norm", "head", and "total". "attention", "mlp", and "norm" are summed over all
    layers ("norm" includes the final norm); "total" is the sum of the other five.
    """
    raise NotImplementedError
