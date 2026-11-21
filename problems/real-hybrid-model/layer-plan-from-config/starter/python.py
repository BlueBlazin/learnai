def layer_plan(config: dict) -> list[dict]:
    """The per-layer plan of a decoder, expanded from its `config.json`.

    config: the parsed file, a dict. The decoder's settings are `config["text_config"]`
        when that key is present and `config` itself otherwise; their "model_type" is
        "qwen3_5_text", "glm_moe_dsa", or "gemma4_text" (anything else: ValueError).
    Returns a list with one dict per layer, in layer order, each with exactly the keys
        "mixer":   str
        "mlp":     "dense" or "sparse"
        "indexer": None, "full", or "shared"
        "rope":    None, or {"type": str, "theta": number, "head_dim": int, "rotary_dim": int}
        "kv":      None, "store", or "reuse"
    The statement has the rules of each family.
    """
    raise NotImplementedError
