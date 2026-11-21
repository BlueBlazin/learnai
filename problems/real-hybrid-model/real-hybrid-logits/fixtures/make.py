"""Regenerate the fixtures of real-hybrid-logits.

    uv run --project python --extra models \
        python problems/real-hybrid-model/real-hybrid-logits/fixtures/make.py

Needs ``Qwen/Qwen3.5-0.8B`` on disk as the kit fetches it (``learnai assets fetch
qwen3.5-0.8b``: the weights, ``config.json``, ``model.safetensors.index.json``, which
``transformers`` needs to find the weight file, and ``tokenizer.json``, which gives the
prompts their ids) and the ``models`` extra. The reference implementation is
``transformers`` (``Qwen3_5ForCausalLM``, float32, CPU, one call per prompt, no cache);
nothing here comes from the kit's forward pass or from ``mlx-lm``, whose float32 logits
for this checkpoint differ from these by tenths (see the editorial).

- ``tiny.npz``: two small models with random weights, for the cases that need no
  download. Each is a ``Qwen3_5ForCausalLM`` built from a tiny ``Qwen3_5TextConfig`` and
  holding the kit's seeded weights ``hybrid_params(cfg, SEED)`` (the module stores the
  block norms, the final norm, and the QK norms as gain - 1, which is what
  ``checkpoint_layout`` hands the solution). The weights are not stored: ``CONFIGS``
  here and in ``tests.py`` are the same objects, and the file records a checksum.
  Stored are the token ids and the full logits.
- ``qwen3.5-0.8b.npz`` and ``prompts.json``: for each prompt (all 12 tokens long, so
  that JAX compiles one set of shapes) the ids, the argmax at every position with the
  margin between the two largest logits, the logits of every position at 64 fixed
  vocabulary columns, and the 8 largest logits of the last position.

One forward pass of the real model takes about 5 s: without the optional fast kernels
``transformers`` runs the delta rule in its reference PyTorch form.
"""

import json
from pathlib import Path

import numpy as np
import torch
import transformers
from tokenizers import Tokenizer
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

from learnai import assets
from learnai.kit import HybridConfig, hybrid_params
from learnai.kit.realmodel import QWEN35
from learnai.problem import find_root

HERE = Path(__file__).parent
PROMPTS = [
    "The capital of France is Paris. The capital of Japan is",
    "<|im_start|>user\nName a color.<|im_end|>\n<|im_start|>assistant\n",
]
LENGTH = 12
COLUMNS = np.sort(np.random.default_rng(251).choice(248320, size=64, replace=False))

SEED = 251
FULL, LINEAR = "full_attention", "linear_attention"
# Keep in step with tests.py. Both have as many key heads as value heads, as the real
# model does, and the same tensor sizes, so the JAX reference compiles once.
CONFIGS = {
    # linear, linear, linear, full: the real model's period and RoPE settings.
    "a": HybridConfig(n_key_heads=4, rope_base=1e7),
    # An explicit schedule that starts with attention, another base, half the head rotating,
    # and an epsilon large enough that every norm shows whether it read cfg.norm_eps.
    "b": HybridConfig(
        n_layers=6, n_key_heads=4, rope_base=1e4, partial_rotary_factor=0.5, norm_eps=1e-2,
        layer_types=(FULL, LINEAR, LINEAR, FULL, LINEAR, FULL),
    ),
}  # fmt: skip
TOKENS = {
    "0": np.array([3, 1, 4, 1, 5, 9]),
    "1": np.array([7, 7, 7, 7, 63, 0]),
}
OFFSET_NORMS = (
    "input_layernorm.weight", "post_attention_layernorm.weight", "q_norm.weight",
    "k_norm.weight", "model.norm.weight",
)  # fmt: skip


def checksum(params: dict[str, np.ndarray]) -> float:
    return float(sum(np.abs(value.astype(np.float64)).sum() for value in params.values()))


def tiny_qwen(cfg: HybridConfig, params: dict[str, np.ndarray]) -> Qwen3_5ForCausalLM:
    """A text-only Qwen3.5 of the kit's small sizes holding the kit's weights."""
    config = Qwen3_5TextConfig(
        vocab_size=cfg.vocab_size, hidden_size=cfg.d_model, intermediate_size=cfg.d_mlp,
        num_hidden_layers=cfg.n_layers, num_attention_heads=cfg.n_heads,
        num_key_value_heads=cfg.n_kv_heads, head_dim=cfg.head_dim, rms_norm_eps=cfg.norm_eps,
        max_position_embeddings=512, tie_word_embeddings=True,
        rope_parameters={
            "rope_type": "default", "rope_theta": cfg.rope_base,
            "partial_rotary_factor": cfg.partial_rotary_factor,
        },
        linear_conv_kernel_dim=cfg.conv_kernel, linear_key_head_dim=cfg.key_head_dim,
        linear_value_head_dim=cfg.value_head_dim, linear_num_key_heads=cfg.n_key_heads,
        linear_num_value_heads=cfg.n_value_heads, layer_types=list(cfg.layers),
    )  # fmt: skip
    model = Qwen3_5ForCausalLM(config).eval().float()
    expected = model.state_dict()
    state = {}
    for name, value in params.items():
        tensor = torch.from_numpy(value.copy())
        # The module computes rms_norm(x) * (1 + weight) for these: store gain - 1.
        state[name] = tensor - 1.0 if name.endswith(OFFSET_NORMS) else tensor
        assert expected[name].shape == state[name].shape, name
    state["lm_head.weight"] = torch.from_numpy(params["model.embed_tokens.weight"].copy())
    assert set(expected) == set(state)
    model.load_state_dict(state)
    return model


def tiny() -> None:
    out: dict[str, np.ndarray] = {
        "transformers_version": np.array(transformers.__version__),
        "seed": np.array(SEED),
    }
    for key, tokens in TOKENS.items():
        out[f"tokens_{key}"] = tokens
    for name, cfg in CONFIGS.items():
        params = hybrid_params(cfg, SEED)
        model = tiny_qwen(cfg, params)
        out[f"checksum_{name}"] = np.array(checksum(params))
        for key, tokens in TOKENS.items():
            with torch.no_grad():
                logits = model(torch.from_numpy(tokens)[None]).logits[0].numpy()
            out[f"logits_{name}_{key}"] = logits.astype(np.float32)
    np.savez_compressed(HERE / "tiny.npz", **out)
    print(f"wrote tiny.npz ({(HERE / 'tiny.npz').stat().st_size} bytes)")
    print("worked example, logits_a_0[5, :5]:", np.round(out["logits_a_0"][5, :5], 6))
    print("worked example, argmax per position:", out["logits_a_0"].argmax(-1))


def qwen() -> None:
    directory = assets.resolve(QWEN35, assets.models_dir(find_root()))
    tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
    model, info = Qwen3_5ForCausalLM.from_pretrained(
        directory, dtype=torch.float32, output_loading_info=True
    )
    assert not info["missing_keys"] and not info["unexpected_keys"], info
    model.eval()
    out: dict[str, np.ndarray] = {"columns": COLUMNS, "revision": np.array(QWEN35.revision)}
    for index, prompt in enumerate(PROMPTS):
        ids = tokenizer.encode(prompt).ids
        assert len(ids) == LENGTH, (prompt, len(ids))
        with torch.no_grad():
            logits = model(torch.tensor([ids])).logits[0].numpy()
        ranked = np.sort(logits, axis=-1)
        top = np.argsort(-logits[-1])[:8]
        out[f"ids_{index}"] = np.array(ids, dtype=np.int32)
        out[f"argmax_{index}"] = logits.argmax(-1).astype(np.int32)
        out[f"margin_{index}"] = (ranked[:, -1] - ranked[:, -2]).astype(np.float32)
        out[f"logits_{index}"] = logits[:, COLUMNS]
        out[f"top_ids_{index}"] = top.astype(np.int32)
        out[f"top_logits_{index}"] = logits[-1, top]
        nxt = tokenizer.decode([int(top[0])])
        print(f"prompt {index}: next token {int(top[0])} {nxt!r}, logit {logits[-1, top[0]]:.3f}, "
              f"largest |logit| {np.abs(logits).max():.1f}, "
              f"margins {np.round(np.sort(out[f'margin_{index}'])[:3], 3)}")  # fmt: skip
    np.savez_compressed(HERE / f"{QWEN35.name}.npz", **out)
    (HERE / "prompts.json").write_text(
        json.dumps(PROMPTS, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {QWEN35.name}.npz ({(HERE / f'{QWEN35.name}.npz').stat().st_size} bytes)")


if __name__ == "__main__":
    tiny()
    qwen()
