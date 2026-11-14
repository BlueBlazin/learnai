"""Regenerate the fixtures of mlx-real-8bit-generate.

    uv run --project python --extra jax --extra mlx --extra models \
        python problems/mlx/mlx-real-8bit-generate/fixtures/make.py

Needs the 8-bit house model on disk (``ws assets fetch granite-4.2-3b-mlx-8bit``, 3.9
GB) and the ``models`` extra. The reference implementation is ``mlx-lm``
(``mlx_lm.utils.load_model``), with the floating parameters converted to float32:
``mlx-lm`` itself runs the checkpoint in bfloat16, whose logits differ by up to about
0.4, enough to flip a near-tie.

- ``tiny-configs.json``: the ``config.json`` of two three-layer quantized Granite
  models with a 256-token vocabulary, for the cases that need no download. Model ``a``
  has the constants of Granite 4.2 3B (multipliers 1, an untied head, 8 bits in groups
  of 64); model ``b`` has other multipliers, a RoPE base of 10,000, another epsilon, and 4
  bits in groups of 32. This file is written by hand below and read back here.
- ``tiny.npz``: for each model, a checksum of its checkpoint, the logits of one prompt,
  and the greedy continuation of three prompts with the margin between the two
  largest logits at every step. The checkpoints are not stored: ``tiny_checkpoint``
  below and its copy in ``tests.py`` draw codes, scales, and biases from the same
  seed.
- ``granite-4.2-3b-mlx-8bit.npz`` and ``prompts.json``: for each prompt its token ids
  (chat prompts are rendered with the model's own template by the ``tokenizers``
  library through ``mlx-lm``), the logits of its last position at 64 fixed columns,
  the 8 largest of them, and the greedy continuation with its margins.
"""

import json
import shutil
import tempfile
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.utils import load_model

from whetstone import assets
from whetstone.kit.realmodel import LARGE_8BIT
from whetstone.problem import find_root

HERE = Path(__file__).parent
COLUMNS = np.sort(np.random.default_rng(180).choice(100352, size=64, replace=False))
MIN_MARGIN = 0.5  # every stored greedy step of the real model is clearer than this
NEW_TOKENS = 8
EOS_ID = 100257  # <|im_end|>

BASE = {
    "model_type": "granite",
    "vocab_size": 256,
    "hidden_size": 128,
    "intermediate_size": 256,
    "num_hidden_layers": 3,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "attention_bias": False,
    "mlp_bias": False,
    "max_position_embeddings": 4096,
    "bos_token_id": 1,
    "eos_token_id": 2,
    "pad_token_id": 2,
}
TINY = {
    "a": {
        "seed": 180,
        "config": {
            **BASE,
            "attention_multiplier": 1 / 64,
            "embedding_multiplier": 1.0,
            "residual_multiplier": 1.0,
            "logits_scaling": 1.0,
            "rope_theta": 1e7,
            "rms_norm_eps": 1e-5,
            "tie_word_embeddings": False,
            "quantization": {"group_size": 64, "bits": 8, "mode": "affine"},
        },
    },
    "b": {
        "seed": 181,
        "config": {
            **BASE,
            "attention_multiplier": 1 / 8,
            "embedding_multiplier": 3.0,
            "residual_multiplier": 0.6,
            "logits_scaling": 2.5,
            "rope_theta": 1e4,
            "rms_norm_eps": 1e-2,
            "tie_word_embeddings": False,
            "quantization": {"group_size": 32, "bits": 4, "mode": "affine"},
        },
    },
}
TINY_PROMPTS = [[1, 17, 200, 45, 9, 130], [1, 77], [1, 5, 5, 5, 250, 31, 8, 99, 100, 101, 64]]
TINY_NEW_TOKENS = 12

REAL_PROMPTS = [
    {"text": "The capital of France is Paris. The capital of Japan is"},
    {"messages": [{"role": "user", "content": "Name the three primary colors."}]},
    {"messages": [{"role": "user", "content": "Reply with exactly one word: yes"}],
     "enable_thinking": False},
]  # fmt: skip


def tiny_checkpoint(seed: int, config: dict) -> dict[str, np.ndarray]:
    """A random quantized Granite checkpoint in the MLX layout. Keep in step with tests.py.

    Every linear layer and the embedding is stored as packed codes (uint32, lowest
    bits first) with one scale and one bias per group; the values are float32 numbers
    that bfloat16 represents exactly, so the file can store them as bfloat16.
    """
    rng = np.random.default_rng(seed)
    d, f, vocab = config["hidden_size"], config["intermediate_size"], config["vocab_size"]
    head = d // config["num_attention_heads"]
    kv = head * config["num_key_value_heads"]
    bits, group = config["quantization"]["bits"], config["quantization"]["group_size"]
    levels = 1 << bits
    code_std = float(np.sqrt((levels * levels - 1) / 12.0))

    def bf16(values: np.ndarray) -> np.ndarray:
        raw = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
        return (raw & np.uint32(0xFFFF0000)).view(np.float32)

    out: dict[str, np.ndarray] = {}

    def quantized(name: str, rows: int, cols: int, std: float) -> None:
        codes = rng.integers(0, levels, (rows, cols)).astype(np.uint64)
        per_word = 32 // bits
        shifts = np.arange(per_word, dtype=np.uint64) * np.uint64(bits)
        words = (codes.reshape(rows, cols // per_word, per_word) << shifts).sum(axis=-1)
        scales = std / code_std * rng.uniform(0.5, 1.5, (rows, cols // group))
        scales = bf16(scales * rng.choice([-1.0, 1.0], scales.shape))
        biases = bf16(-scales * (levels - 1) * rng.uniform(0.35, 0.65, scales.shape))
        out[name + ".weight"] = words.astype(np.uint32)
        out[name + ".scales"] = scales
        out[name + ".biases"] = biases

    def gain(name: str) -> None:
        out[name + ".weight"] = bf16(rng.uniform(0.5, 1.5, d))

    quantized("model.embed_tokens", vocab, d, 1.0)
    for i in range(config["num_hidden_layers"]):
        p = f"model.layers.{i}."
        gain(p + "input_layernorm")
        quantized(p + "self_attn.q_proj", d, d, 0.5)
        quantized(p + "self_attn.k_proj", kv, d, 0.5)
        quantized(p + "self_attn.v_proj", kv, d, 0.1)
        quantized(p + "self_attn.o_proj", d, d, 0.1)
        gain(p + "post_attention_layernorm")
        quantized(p + "mlp.gate_proj", f, d, 0.1)
        quantized(p + "mlp.up_proj", f, d, 0.1)
        quantized(p + "mlp.down_proj", d, f, 0.1)
    gain("model.norm")
    if not config["tie_word_embeddings"]:
        quantized("lm_head", vocab, d, 0.3)
    return out


def checksum(tensors: dict[str, np.ndarray]) -> float:
    return float(sum(np.abs(t.astype(np.float64)).sum() for t in tensors.values()))


def write_checkpoint(directory: Path, config: dict, tensors: dict[str, np.ndarray]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    stored = {
        name: mx.array(t) if t.dtype == np.uint32 else mx.array(t).astype(mx.bfloat16)
        for name, t in tensors.items()
    }
    mx.save_safetensors(str(directory / "model.safetensors"), stored, metadata={"format": "mlx"})


def greedy(
    model, ids: list[int], n: int, eos_id: int | None = None
) -> tuple[list[int], list[float], np.ndarray]:
    """Up to n greedy tokens without a cache (stopping after `eos_id`), the margin of
    each, and the prompt's logits."""
    sequence, tokens, margins, first = list(ids), [], [], None
    for _ in range(n):
        logits = np.array(model(mx.array([sequence]))[0].astype(mx.float32))
        first = logits if first is None else first
        ranked = np.sort(logits[-1])
        margins.append(float(ranked[-1] - ranked[-2]))
        tokens.append(int(logits[-1].argmax()))
        sequence.append(tokens[-1])
        if tokens[-1] == eos_id:
            break
    return tokens, margins, first


def tiny() -> None:
    (HERE / "tiny-configs.json").write_text(json.dumps(TINY, indent=2) + "\n", encoding="utf-8")
    out: dict[str, np.ndarray] = {}
    scratch = Path(tempfile.mkdtemp(prefix="whetstone-make-"))
    try:
        for name, spec in TINY.items():
            tensors = tiny_checkpoint(spec["seed"], spec["config"])
            write_checkpoint(scratch / name, spec["config"], tensors)
            model, _ = load_model(scratch / name)
            model.set_dtype(mx.float32)
            out[f"checksum_{name}"] = np.array(checksum(tensors))
            for index, prompt in enumerate(TINY_PROMPTS):
                tokens, margins, logits = greedy(model, prompt, TINY_NEW_TOKENS)
                out[f"prompt_{index}"] = np.array(prompt, dtype=np.int32)
                out[f"tokens_{name}_{index}"] = np.array(tokens, dtype=np.int32)
                out[f"margins_{name}_{index}"] = np.array(margins, dtype=np.float32)
                if index == 0:
                    out[f"logits_{name}"] = logits.astype(np.float32)
                print(f"tiny {name}, prompt {index}: {tokens}, smallest margin {min(margins):.3f}")
            print(
                f"tiny {name}: logits[-1, :4] of prompt 0 =",
                np.round(out[f"logits_{name}"][-1, :4], 6),
            )
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    np.savez_compressed(HERE / "tiny.npz", **out)
    print(f"wrote tiny.npz ({(HERE / 'tiny.npz').stat().st_size} bytes)")


def granite() -> None:
    directory = assets.resolve(LARGE_8BIT, assets.models_dir(find_root()))
    model, tokenizer = load(str(directory))
    model.set_dtype(mx.float32)
    out: dict[str, np.ndarray] = {"columns": COLUMNS, "revision": np.array(LARGE_8BIT.revision)}
    described = []
    for index, prompt in enumerate(REAL_PROMPTS):
        if "text" in prompt:
            ids = tokenizer.encode(prompt["text"])
        else:
            ids = tokenizer.apply_chat_template(
                prompt["messages"],
                add_generation_prompt=True,
                enable_thinking=prompt.get("enable_thinking", True),
            )
        tokens, margins, logits = greedy(model, ids, NEW_TOKENS, EOS_ID)
        assert min(margins) > MIN_MARGIN, (index, margins)
        top = np.argsort(-logits[-1])[:8]
        out[f"ids_{index}"] = np.array(ids, dtype=np.int32)
        out[f"tokens_{index}"] = np.array(tokens, dtype=np.int32)
        out[f"margins_{index}"] = np.array(margins, dtype=np.float32)
        out[f"logits_{index}"] = logits[-1, COLUMNS]
        out[f"top_ids_{index}"] = top.astype(np.int32)
        out[f"top_logits_{index}"] = logits[-1, top]
        described.append(
            {"prompt": tokenizer.decode(ids), "continuation": tokenizer.decode(tokens)}
        )
        print(f"prompt {index}: {len(ids)} tokens -> {tokens} {tokenizer.decode(tokens)!r}, "
              f"smallest margin {min(margins):.3f}")  # fmt: skip
    np.savez_compressed(HERE / "granite-4.2-3b-mlx-8bit.npz", **out)
    (HERE / "prompts.json").write_text(
        json.dumps(described, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    size = (HERE / "granite-4.2-3b-mlx-8bit.npz").stat().st_size
    print(f"wrote granite-4.2-3b-mlx-8bit.npz ({size} bytes)")


if __name__ == "__main__":
    tiny()
    granite()
