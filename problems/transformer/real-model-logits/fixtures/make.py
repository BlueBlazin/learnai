"""Regenerate the fixtures of real-model-logits.

    uv run --project python --extra jax --extra mlx --extra models \
        python problems/transformer/real-model-logits/fixtures/make.py

Needs the small house model on disk (``ws assets fetch granite-4.0-350m``) and the
``models`` extra. The reference implementation is ``transformers`` (float32, CPU);
token ids come from its tokenizer.

- ``tiny.npz`` and ``tiny-configs.json``: two three-layer Granite models with random
  weights and a 64-token vocabulary, for the cases that need no download. Model ``a``
  has Granite 4.0's multipliers and a tied head; model ``b`` has other multipliers, a
  RoPE base of 10,000, and an untied head. Both are 128 wide with 4 query heads on 2
  KV heads of width 32, so every such case has one set of shapes. The weights are not
  stored: ``tiny_weights`` below and its copy in ``tests.py`` draw them from the same
  seed, and the file records a checksum. Stored are the token ids and the full logits.
- ``granite-4.0-350m.npz`` and ``prompts.json``: for each prompt (all 12 tokens long,
  so that JAX compiles one set of shapes) the ids, the argmax at every position with
  the margin between the two largest logits, the logits of every position at 64 fixed
  vocabulary columns, and the 8 largest logits of the last position.
"""

import json
from pathlib import Path

import numpy as np
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GraniteMoeHybridConfig,
    GraniteMoeHybridForCausalLM,
)

from whetstone import assets
from whetstone.kit.realmodel import SMALL
from whetstone.problem import find_root

HERE = Path(__file__).parent
PROMPTS = [
    "The capital of France is Paris. The capital of Japan is",
    "def square(x):\n    return x * x\n\nprint(",
    "<|start_of_role|>user<|end_of_role|>Name a color.<|end_of_text|>\n"
    + "<|start_of_role|>assistant<|end_of_role|>",
    "🗡️ blades; 東京 is in the",
]
LENGTH = 12
COLUMNS = np.sort(np.random.default_rng(190).choice(100352, size=64, replace=False))

TINY = {"vocab": 64, "d_model": 128, "n_heads": 4, "n_kv_heads": 2, "d_ff": 96, "n_layers": 3}
VARIANTS = {
    "a": {"attention": 1 / 64, "embedding": 12.0, "residual": 0.263, "logits": 4.0,
          "theta": 1e7, "eps": 1e-5, "tied": True},
    "b": {"attention": 1 / 8, "embedding": 3.0, "residual": 0.6, "logits": 2.5,
          "theta": 1e4, "eps": 1e-2, "tied": False},
}  # fmt: skip


def tiny_weights(seed: int, tied: bool) -> dict[str, np.ndarray]:
    """Random weights with Granite 4.0's names and layout. Keep in step with tests.py."""
    rng = np.random.default_rng(seed)
    d, f, vocab = TINY["d_model"], TINY["d_ff"], TINY["vocab"]
    q_width = d
    kv_width = d * TINY["n_kv_heads"] // TINY["n_heads"]

    def normal(rows: int, cols: int, std: float) -> np.ndarray:
        return (rng.standard_normal((rows, cols)) * std).astype(np.float32)

    def gain() -> np.ndarray:
        return rng.uniform(0.5, 1.5, d).astype(np.float32)

    weights = {"model.embed_tokens.weight": normal(vocab, d, 1.0)}
    for i in range(TINY["n_layers"]):
        p = f"model.layers.{i}."
        weights[p + "input_layernorm.weight"] = gain()
        weights[p + "self_attn.q_proj.weight"] = normal(q_width, d, 0.3)
        weights[p + "self_attn.k_proj.weight"] = normal(kv_width, d, 0.3)
        weights[p + "self_attn.v_proj.weight"] = normal(kv_width, d, 0.1)
        weights[p + "self_attn.o_proj.weight"] = normal(d, q_width, 0.1)
        weights[p + "post_attention_layernorm.weight"] = gain()
        weights[p + "shared_mlp.input_linear.weight"] = normal(2 * f, d, 0.1)
        weights[p + "shared_mlp.output_linear.weight"] = normal(d, f, 0.1)
    weights["model.norm.weight"] = gain()
    if not tied:
        weights["lm_head.weight"] = normal(vocab, d, 1.0)
    return weights


def checksum(weights: dict[str, np.ndarray]) -> float:
    return float(sum(np.abs(w.astype(np.float64)).sum() for w in weights.values()))


def tiny() -> None:
    rng = np.random.default_rng(191)
    tokens = {
        "tokens_0": rng.integers(0, TINY["vocab"], (2, 6)),
        "tokens_1": np.array([[7, 7, 7, 7, 7, 7], [9, 4, 9, 4, 9, 63]]),
    }
    out: dict[str, np.ndarray] = dict(tokens)
    configs = {}
    for seed, (name, v) in enumerate(VARIANTS.items(), start=190):
        config = GraniteMoeHybridConfig(
            vocab_size=TINY["vocab"],
            hidden_size=TINY["d_model"],
            intermediate_size=TINY["d_ff"],
            shared_intermediate_size=TINY["d_ff"],
            num_hidden_layers=TINY["n_layers"],
            num_attention_heads=TINY["n_heads"],
            num_key_value_heads=TINY["n_kv_heads"],
            layer_types=["attention"] * TINY["n_layers"],
            num_local_experts=0,
            num_experts_per_tok=0,
            attention_multiplier=v["attention"],
            embedding_multiplier=v["embedding"],
            residual_multiplier=v["residual"],
            logits_scaling=v["logits"],
            position_embedding_type="rope",
            rope_parameters={"rope_theta": v["theta"], "rope_type": "default"},
            rms_norm_eps=v["eps"],
            tie_word_embeddings=v["tied"],
            max_position_embeddings=4096,
            pad_token_id=None,
            bos_token_id=1,
            eos_token_id=1,
        )
        model = GraniteMoeHybridForCausalLM(config).eval()
        # A config without the two rope entries silently builds a model without RoPE.
        assert abs(float(model.model.rotary_emb.inv_freq[1]) - v["theta"] ** (-1 / 16)) < 1e-6
        weights = tiny_weights(seed, v["tied"])
        state = {key: torch.from_numpy(array) for key, array in weights.items()}
        if v["tied"]:
            state["lm_head.weight"] = state["model.embed_tokens.weight"]
        model.load_state_dict(state, strict=True)
        assert (model.lm_head.weight is model.model.embed_tokens.weight) == v["tied"]
        for key, ids in tokens.items():
            with torch.no_grad():
                logits = model(torch.from_numpy(ids)).logits.numpy()
            out[f"logits_{name}_{key[-1]}"] = logits.astype(np.float32)
        out[f"checksum_{name}"] = np.array(checksum(weights))
        # The same facts in the key style of Granite 4.0's own config.json, which is
        # what the kit's RealConfig.from_json reads (transformers 5 re-serializes
        # layer_types as "full_attention" and rope_theta inside rope_parameters).
        configs[name] = {
            "seed": seed,
            "config": {
                "model_type": "granitemoehybrid",
                "layer_types": ["attention"] * TINY["n_layers"],
                "num_local_experts": 0,
                "vocab_size": TINY["vocab"],
                "hidden_size": TINY["d_model"],
                "intermediate_size": TINY["d_ff"],
                "shared_intermediate_size": TINY["d_ff"],
                "num_hidden_layers": TINY["n_layers"],
                "num_attention_heads": TINY["n_heads"],
                "num_key_value_heads": TINY["n_kv_heads"],
                "attention_multiplier": config.attention_multiplier,
                "embedding_multiplier": config.embedding_multiplier,
                "residual_multiplier": config.residual_multiplier,
                "logits_scaling": config.logits_scaling,
                "rope_theta": config.rope_parameters["rope_theta"],
                "rms_norm_eps": config.rms_norm_eps,
                "tie_word_embeddings": config.tie_word_embeddings,
                "max_position_embeddings": config.max_position_embeddings,
                "bos_token_id": 1,
                "eos_token_id": 1,
                "pad_token_id": None,
            },
        }
    np.savez_compressed(HERE / "tiny.npz", **out)
    (HERE / "tiny-configs.json").write_text(json.dumps(configs, indent=2) + "\n", "utf-8")
    print(f"wrote tiny.npz ({(HERE / 'tiny.npz').stat().st_size} bytes)")
    print("worked example, logits_a_0[0, 5, :5]:", np.round(out["logits_a_0"][0, 5, :5], 4))
    print("tokens_0:", tokens["tokens_0"].tolist())


def granite() -> None:
    directory = assets.resolve(SMALL, assets.models_dir(find_root()))
    tokenizer = AutoTokenizer.from_pretrained(directory)
    model = AutoModelForCausalLM.from_pretrained(directory, dtype=torch.float32).eval()
    out: dict[str, np.ndarray] = {"columns": COLUMNS, "revision": np.array(SMALL.revision)}
    for index, prompt in enumerate(PROMPTS):
        ids = tokenizer(prompt, return_tensors="pt").input_ids
        assert ids.shape == (1, LENGTH), (prompt, ids.shape)
        with torch.no_grad():
            logits = model(ids).logits[0].numpy()
        ranked = np.sort(logits, axis=-1)
        top = np.argsort(-logits[-1])[:8]
        out[f"ids_{index}"] = ids[0].numpy().astype(np.int32)
        out[f"argmax_{index}"] = logits.argmax(-1).astype(np.int32)
        out[f"margin_{index}"] = (ranked[:, -1] - ranked[:, -2]).astype(np.float32)
        out[f"logits_{index}"] = logits[:, COLUMNS]
        out[f"top_ids_{index}"] = top.astype(np.int32)
        out[f"top_logits_{index}"] = logits[-1, top]
        nxt = tokenizer.decode([int(top[0])])
        print(f"prompt {index}: next token {nxt!r}, logit {logits[-1, top[0]]:.3f}, "
              f"smallest margin {out[f'margin_{index}'].min():.3f}")  # fmt: skip
    np.savez_compressed(HERE / "granite-4.0-350m.npz", **out)
    (HERE / "prompts.json").write_text(
        json.dumps(PROMPTS, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote granite-4.0-350m.npz ({(HERE / 'granite-4.0-350m.npz').stat().st_size} bytes)")


if __name__ == "__main__":
    tiny()
    granite()
