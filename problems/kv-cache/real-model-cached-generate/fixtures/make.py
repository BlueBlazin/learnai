"""Regenerate the fixtures of real-model-cached-generate.

    uv run --project python --extra jax --extra mlx --extra models \
        python problems/kv-cache/real-model-cached-generate/fixtures/make.py

Needs the small house model on disk (``ws assets fetch granite-4.0-350m``) and the
``models`` extra. The reference implementation is ``transformers`` (float32, CPU):
greedy ``generate`` with its own cache for the tokens, and plain forward passes over
whole sequences, without a cache, for the logits.

- ``tiny.npz`` and ``tiny-configs.json``: two three-layer Granite models with random
  weights and a 64-token vocabulary, for the cases that need no download (the models
  of ``real-model-logits``: ``a`` has Granite 4.0's constants and a tied head, ``b``
  other constants, RoPE base 10,000, and an untied head). The weights are not stored:
  ``tiny_weights`` below and its copy in ``tests.py`` draw them from the same seed,
  and the file records a checksum. Stored per model and prompt: the prompt, the
  greedy tokens, the full logits that chose each of them, and the logits after a
  second, different token fed in place of the first generated one (a branch).
- ``granite-4.0-350m.npz`` and ``prompts.json``: the same for the real model, with
  the logits at 64 fixed vocabulary columns; two prompts of 12 tokens and one of 768
  tokens, which is what the speed case decodes after.
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
COLUMNS = np.sort(np.random.default_rng(160).choice(100352, size=64, replace=False))
PROMPTS = [
    "A B C D E F G H I J K L",
    "for i in range(10):\n    print(i)\n\nfor",
]
SHORT, NEW = 12, 12
LONG, LONG_NEW = 768, 6
LONG_TEXT = (
    "A language model writes one token at a time. To choose the next token it runs "
    "every layer of the network over the text so far, and each attention layer compares "
    "the newest token with all the tokens before it. Done naively, the work for one "
    "token grows with the length of the text, and the work for a whole reply grows with "
    "its square. The key-value cache removes the repetition. The keys and values that "
    "attention computes for a token depend only on that token and the ones before it, "
    "so they never change once they are computed. A runtime stores them, layer by "
    "layer, and each new token computes only its own query, key, and value and reads "
    "the rest from the cache. The price is memory: the cache holds two vectors per "
    "token, per layer, per key-value head, and for a long conversation that is "
    "gigabytes. Grouped-query attention makes it smaller by sharing each key-value head "
    "among several query heads, which is why the cache is kept at the width of the "
    "key-value heads and expanded only when it is used. Position information has to "
    "survive the cache as well. With rotary position embeddings a key is rotated by an "
    "angle that depends on its position, once, when it is computed, and stored rotated. "
    "A new token is rotated for its own position, which is the number of tokens already "
    "in the cache, not zero. Runtimes differ in how they store the cache. Some append a "
    "slice per token, some reserve room for the whole context in advance, and some keep "
    "it in fixed-size blocks so that many requests can share one pool of memory. All of "
    "them give the same tokens as the naive loop, only faster. This is why the first "
    "token of a reply is slow and the rest are quick: reading the prompt fills the "
    "cache for hundreds of tokens at once, and after that each step handles a single "
    "token. It is also why a server can keep the cache of a conversation and continue "
    "it when the next message arrives, instead of reading everything again. The rest "
    "of this text is padding for the test: the quick brown fox jumps over the lazy dog, "
    "and the small model is asked to continue from here in its own words, one token "
    "after another, each chosen as the most likely one. In summary, the key-value cache"
)
#: The long prompt ends in something predictable, so the greedy tokens have clear margins.
LONG_SUFFIX = "\n\nThe alphabet: A B C D E F G H I J K L"
#: Greedy tokens must win by at least this much, so float noise cannot change them.
MARGIN, TINY_MARGIN = 0.25, 0.02

TINY = {"vocab": 64, "d_model": 128, "n_heads": 4, "n_kv_heads": 2, "d_ff": 96, "n_layers": 3}
TINY_NEW = 8
VARIANTS = {
    "a": {"attention": 1 / 64, "embedding": 12.0, "residual": 0.263, "logits": 4.0,
          "theta": 1e7, "eps": 1e-5, "tied": True, "seed": 208, "embed_std": 0.1},
    "b": {"attention": 1 / 8, "embedding": 3.0, "residual": 0.6, "logits": 2.5,
          "theta": 1e4, "eps": 1e-2, "tied": False, "seed": 191, "embed_std": 1.0},
}  # fmt: skip


def tiny_weights(seed: int, tied: bool, embed_std: float) -> dict[str, np.ndarray]:
    """Random weights with Granite 4.0's names and layout. Keep in step with tests.py.

    The branch matrices are large and the embedding of the tied model is small, so
    that attention and the MLP move the stream: otherwise a tied head only ever
    repeats the last token.
    """
    rng = np.random.default_rng(seed)
    d, f, vocab = TINY["d_model"], TINY["d_ff"], TINY["vocab"]
    q_width = d
    kv_width = d * TINY["n_kv_heads"] // TINY["n_heads"]

    def normal(rows: int, cols: int, std: float) -> np.ndarray:
        return (rng.standard_normal((rows, cols)) * std).astype(np.float32)

    def gain() -> np.ndarray:
        return rng.uniform(0.5, 1.5, d).astype(np.float32)

    weights = {"model.embed_tokens.weight": normal(vocab, d, embed_std)}
    for i in range(TINY["n_layers"]):
        p = f"model.layers.{i}."
        weights[p + "input_layernorm.weight"] = gain()
        weights[p + "self_attn.q_proj.weight"] = normal(q_width, d, 0.3)
        weights[p + "self_attn.k_proj.weight"] = normal(kv_width, d, 0.3)
        weights[p + "self_attn.v_proj.weight"] = normal(kv_width, d, 0.3)
        weights[p + "self_attn.o_proj.weight"] = normal(d, q_width, 0.3)
        weights[p + "post_attention_layernorm.weight"] = gain()
        weights[p + "shared_mlp.input_linear.weight"] = normal(2 * f, d, 0.2)
        weights[p + "shared_mlp.output_linear.weight"] = normal(d, f, 0.3)
    weights["model.norm.weight"] = gain()
    if not tied:
        weights["lm_head.weight"] = normal(vocab, d, 1.0)
    return weights


def checksum(weights: dict[str, np.ndarray]) -> float:
    return float(sum(np.abs(w.astype(np.float64)).sum() for w in weights.values()))


def trace(model, prompt: list[int], new: int, margin: float, stop: int | None) -> dict:
    """Greedy tokens of `generate`, and from uncached forward passes: the logits that
    chose each token and the logits of a branch (another token in place of the first)."""
    with torch.no_grad():
        out = model.generate(torch.tensor([prompt]), max_new_tokens=new, do_sample=False,
                             pad_token_id=0, eos_token_id=None)  # fmt: skip
        tokens = out[0, len(prompt) :].tolist()
        assert len(tokens) == new and (stop is None or stop not in tokens[:-1])
        # Row len(prompt) - 1 + i of one pass over prompt + tokens chose token i.
        full = model(torch.tensor([prompt + tokens[:-1]])).logits[0].numpy()
        logits = full[len(prompt) - 1 :]
        ranked = np.sort(logits, axis=-1)
        margins = ranked[:, -1] - ranked[:, -2]
        assert (logits.argmax(-1) == np.array(tokens)).all()
        assert margins.min() > margin, margins
        # The branch: the runner-up of the first step instead of the winner.
        other = int(np.argsort(-logits[0])[1])
        branch = model(torch.tensor([[*prompt, other]])).logits[0, -1].numpy()
    return {"prompt": prompt, "tokens": tokens, "logits": logits.astype(np.float32),
            "margin": float(margins.min()), "other": other, "branch": branch}  # fmt: skip


def tiny() -> None:
    rng = np.random.default_rng(161)
    prompts = [[int(t) for t in rng.integers(0, TINY["vocab"], 6)] for _ in range(2)] + [[41]]
    out: dict[str, np.ndarray] = {}
    configs = {}
    for name, v in VARIANTS.items():
        seed = v["seed"]
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
        weights = tiny_weights(seed, v["tied"], v["embed_std"])
        state = {key: torch.from_numpy(array) for key, array in weights.items()}
        if v["tied"]:
            state["lm_head.weight"] = state["model.embed_tokens.weight"]
        model.load_state_dict(state, strict=True)
        for index, prompt in enumerate(prompts):
            t = trace(model, prompt, TINY_NEW, TINY_MARGIN, None)
            for key, value in t.items():
                if key != "margin":
                    out[f"{key}_{name}_{index}"] = np.asarray(value)
            print(f"tiny {name}, prompt {prompt}: tokens {t['tokens']}, margin {t['margin']:.3f}")
        out[f"checksum_{name}"] = np.array(checksum(weights))
        # The same facts in the key style of Granite 4.0's own config.json, which is
        # what the kit's RealConfig.from_json reads (transformers 5 re-serializes
        # layer_types as "full_attention" and rope_theta inside rope_parameters).
        configs[name] = {
            "seed": seed,
            "embed_std": v["embed_std"],
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


def granite() -> None:
    directory = assets.resolve(SMALL, assets.models_dir(find_root()))
    tokenizer = AutoTokenizer.from_pretrained(directory)
    model = AutoModelForCausalLM.from_pretrained(directory, dtype=torch.float32).eval()
    stop = tokenizer.eos_token_id
    out: dict[str, np.ndarray] = {"columns": COLUMNS, "revision": np.array(SMALL.revision)}
    texts = []
    suffix = tokenizer(LONG_SUFFIX).input_ids
    # The text twice over, cut to length, then the predictable ending.
    long_ids = (tokenizer(LONG_TEXT).input_ids * 2)[: LONG - len(suffix)] + suffix
    plan = [(tokenizer(p).input_ids, NEW, SHORT) for p in PROMPTS] + [(long_ids, LONG_NEW, LONG)]
    for index, (prompt, new, length) in enumerate(plan):
        assert len(prompt) == length, (index, len(prompt))
        t = trace(model, prompt, new, MARGIN, stop)
        out[f"prompt_{index}"] = np.array(prompt, dtype=np.int32)
        out[f"tokens_{index}"] = np.array(t["tokens"], dtype=np.int32)
        out[f"logits_{index}"] = t["logits"][:, COLUMNS]
        out[f"other_{index}"] = np.array(t["other"], dtype=np.int32)
        out[f"branch_{index}"] = t["branch"][COLUMNS]
        texts.append({"prompt": tokenizer.decode(prompt), "reply": tokenizer.decode(t["tokens"])})
        print(f"prompt {index} ({length} tokens): {texts[-1]['reply']!r}, margin {t['margin']:.3f}")
    np.savez_compressed(HERE / "granite-4.0-350m.npz", **out)
    (HERE / "prompts.json").write_text(
        json.dumps(texts, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote granite-4.0-350m.npz ({(HERE / 'granite-4.0-350m.npz').stat().st_size} bytes)")


if __name__ == "__main__":
    tiny()
    granite()
