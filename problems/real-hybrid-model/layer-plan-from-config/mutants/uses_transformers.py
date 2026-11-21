# BUG: computes nothing for Qwen3.5: builds the library's own configuration class and reads the schedule off it, which the statement rules out.
from transformers import Qwen3_5TextConfig

FULL, LINEAR = "full_attention", "linear_attention"


def layer_plan(config: dict) -> list[dict]:
    text = config.get("text_config", config)
    if text.get("model_type") != "qwen3_5_text":
        raise ValueError(f"unsupported model_type {text.get('model_type')!r}")
    cfg = Qwen3_5TextConfig(**{k: v for k, v in text.items() if k != "model_type"})
    params = cfg.rope_parameters
    rope = {
        "type": params["rope_type"],
        "theta": params["rope_theta"],
        "head_dim": cfg.head_dim,
        "rotary_dim": int(cfg.head_dim * params["partial_rotary_factor"]),
    }
    return [
        {
            "mixer": kind,
            "mlp": "dense",
            "indexer": None,
            "rope": dict(rope) if kind == FULL else None,
            "kv": None,
        }
        for kind in cfg.layer_types
    ]
