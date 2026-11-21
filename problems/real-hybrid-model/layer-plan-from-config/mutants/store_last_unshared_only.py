# BUG: marks only the last layer before the shared ones as `store`; every layer type needs its own producer, the last unshared layer of that type.
FULL, LINEAR = "full_attention", "linear_attention"


def layer_plan(config: dict) -> list[dict]:
    # A multimodal file keeps the decoder's settings one level down.
    text = config.get("text_config") or config
    family = text.get("model_type")
    if family == "qwen3_5_text":
        return _qwen(text)
    if family == "glm_moe_dsa":
        return _glm(text)
    if family == "gemma4_text":
        return _gemma(text)
    raise ValueError(f"unsupported model_type {family!r}")


def _entry(mixer, mlp="dense", indexer=None, rope=None, kv=None) -> dict:
    return {"mixer": mixer, "mlp": mlp, "indexer": indexer, "rope": rope, "kv": kv}


def _rope(kind: str, theta: float, head_dim: int, rotary_dim: int) -> dict:
    return {"type": kind, "theta": theta, "head_dim": head_dim, "rotary_dim": rotary_dim}


def _qwen(text: dict) -> list[dict]:
    n = text["num_hidden_layers"]
    mixers = text.get("layer_types")
    if mixers is None:
        # Counted from 1: the last layer of every group of `interval` attends.
        interval = text.get("full_attention_interval", 4)
        mixers = [FULL if (i + 1) % interval == 0 else LINEAR for i in range(n)]
    params = text["rope_parameters"]
    # This family's own default: a quarter of the head rotates, not all of it.
    factor = params.get("partial_rotary_factor", 0.25)
    head = text["head_dim"]
    rope = _rope(params["rope_type"], params["rope_theta"], head, int(head * factor))
    # A linear-attention layer is a recurrence: it is given no positions at all.
    return [_entry(mixer, rope=dict(rope) if mixer == FULL else None) for mixer in mixers]


def _glm(text: dict) -> list[dict]:
    n = text["num_hidden_layers"]
    mlps = text.get("mlp_layer_types")
    if mlps is None:
        dense = text.get("first_k_dense_replace", 3)
        mlps = ["dense" if i < dense else "sparse" for i in range(n)]
    indexers = text.get("indexer_types")
    if indexers is None:
        pattern = text.get("index_topk_pattern")
        if pattern is not None:
            letters = {"F": "full", "S": "shared"}
            indexers = [letters[c] for c in pattern] if isinstance(pattern, str) else list(pattern)
        else:
            freq = max(text.get("index_topk_freq", 1), 1)
            offset = text.get("index_skip_topk_offset", 2)
            # The clamp makes every layer up to `offset - 1` a full layer.
            indexers = ["full" if max(i - offset + 1, 0) % freq == 0 else "shared" for i in range(n)]
    params = text["rope_parameters"]
    # A head is a part without positions and a part that rotates; the file's own
    # `head_dim` key is neither the sum nor, in general, the rotating part.
    rotary = text["qk_rope_head_dim"]
    rope = _rope(params["rope_type"], params["rope_theta"], text["qk_nope_head_dim"] + rotary, rotary)
    return [_entry("indexed_attention", mlps[i], indexers[i], dict(rope)) for i in range(n)]


def _gemma(text: dict) -> list[dict]:
    kinds = text["layer_types"]
    n = len(kinds)
    shared = text.get("num_kv_shared_layers", 0)
    kv = [None] * n
    if shared:
        first = n - shared  # layers first .. n - 1 read keys and values, they make none
        for i in range(first, n):
            kv[i] = "reuse"
        kv[first - 1] = "store"  # the layer right before the shared ones
    plan = []
    for kind, mode in zip(kinds, kv, strict=True):
        params = text["rope_parameters"][kind]
        head = text.get("global_head_dim", 512) if kind == FULL else text["head_dim"]
        if params["rope_type"] == "proportional":
            rotary = 2 * int(params.get("partial_rotary_factor", 1.0) * head // 2)
        else:
            rotary = head
        rope = _rope(params["rope_type"], params["rope_theta"], head, rotary)
        plan.append(_entry(kind, rope=rope, kv=mode))
    return plan
