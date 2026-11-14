"""Regenerate the fixtures of real-model-block.

    uv run --project python --extra jax --extra mlx --extra models \
        python problems/transformer/real-model-block/fixtures/make.py

Needs the small house model on disk (``ws assets fetch granite-4.0-350m``) and the
``models`` extra. The reference implementation is ``transformers`` (float32, CPU):
every expected hidden state is the output of one of its decoder layers, captured with
a forward hook. Two files are written.

- ``tiny.npz``: a three-layer Granite with random weights, small enough that the cases
  using it need no download (width 256: 4 query heads on 2 KV heads of width 64, a
  fused SwiGLU of width 192). The weights are not stored: ``tiny_weights`` below and
  the copy of it in ``tests.py`` draw them from the same seed, and the file records
  their checksum. Stored are inputs, positions, and the full output of each layer.
- ``granite-4.0-350m.npz``: for one prompt, the hidden states entering layers 1 and 13
  (in full: they are inputs of the test), and 64 fixed columns of the states leaving
  layers 0, 1, and 13; plus layer 13 run on the same input at positions with gaps.
"""

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
PROMPT = "The whetstone sharpens the blade, and the blade"
COLUMNS = np.sort(np.random.default_rng(180).choice(1024, size=64, replace=False))
GAPPED = np.array([0, 1, 2, 5, 6, 7, 20, 21, 300, 301, 302, 2000])

TINY = {"d_model": 256, "n_heads": 4, "n_kv_heads": 2, "d_ff": 192, "n_layers": 3}


def tiny_weights(seed: int = 180) -> dict[str, np.ndarray]:
    """Random weights with Granite 4.0's names and layout. Keep in step with tests.py."""
    rng = np.random.default_rng(seed)
    d, f = TINY["d_model"], TINY["d_ff"]

    def normal(rows: int, cols: int, std: float) -> np.ndarray:
        return (rng.standard_normal((rows, cols)) * std).astype(np.float32)

    weights = {}
    for i in range(TINY["n_layers"]):
        p = f"model.layers.{i}."
        weights[p + "input_layernorm.weight"] = rng.uniform(0.5, 1.5, d).astype(np.float32)
        weights[p + "self_attn.q_proj.weight"] = normal(TINY["n_heads"] * 64, d, 0.2)
        weights[p + "self_attn.k_proj.weight"] = normal(TINY["n_kv_heads"] * 64, d, 0.2)
        weights[p + "self_attn.v_proj.weight"] = normal(TINY["n_kv_heads"] * 64, d, 0.06)
        weights[p + "self_attn.o_proj.weight"] = normal(d, TINY["n_heads"] * 64, 0.06)
        weights[p + "post_attention_layernorm.weight"] = rng.uniform(0.5, 1.5, d).astype(np.float32)
        weights[p + "shared_mlp.input_linear.weight"] = normal(2 * f, d, 0.08)
        weights[p + "shared_mlp.output_linear.weight"] = normal(d, f, 0.08)
    return weights


def checksum(weights: dict[str, np.ndarray]) -> float:
    return float(sum(np.abs(w.astype(np.float64)).sum() for w in weights.values()))


def run_layer(model, layer: int, x: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """The output of decoder layer `layer` of `model` for the input `x` (B, T, D) at the
    given positions: a normal forward pass in which a hook swaps the layer's input."""
    captured = {}
    block = model.model.layers[layer]

    def swap_input(module, args, kwargs):
        if args:
            return (torch.from_numpy(x), *args[1:]), kwargs
        return args, {**kwargs, "hidden_states": torch.from_numpy(x)}

    def capture(module, args, kwargs, output):
        captured["out"] = (output[0] if isinstance(output, tuple) else output).detach().numpy()

    handles = [
        block.register_forward_pre_hook(swap_input, with_kwargs=True),
        block.register_forward_hook(capture, with_kwargs=True),
    ]
    try:
        with torch.no_grad():
            model(
                input_ids=torch.zeros(x.shape[:2], dtype=torch.long),
                position_ids=torch.from_numpy(np.broadcast_to(positions, x.shape[:2]).copy()),
            )
    finally:
        for handle in handles:
            handle.remove()
    return captured["out"].astype(np.float32)


def tiny() -> None:
    config = GraniteMoeHybridConfig(
        vocab_size=64,
        hidden_size=TINY["d_model"],
        intermediate_size=TINY["d_ff"],
        shared_intermediate_size=TINY["d_ff"],
        num_hidden_layers=TINY["n_layers"],
        num_attention_heads=TINY["n_heads"],
        num_key_value_heads=TINY["n_kv_heads"],
        layer_types=["attention"] * TINY["n_layers"],
        num_local_experts=0,
        num_experts_per_tok=0,
        attention_multiplier=1 / 64,
        embedding_multiplier=12.0,
        residual_multiplier=0.263,
        logits_scaling=4.0,
        position_embedding_type="rope",
        rope_parameters={"rope_theta": 10_000_000.0, "rope_type": "default"},
        rms_norm_eps=1e-5,
        tie_word_embeddings=True,
        max_position_embeddings=4096,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=1,
    )
    model = GraniteMoeHybridForCausalLM(config).eval()
    # A config without these two silently builds a model with no rotary embedding.
    assert abs(float(model.model.rotary_emb.inv_freq[1]) - 1e7 ** (-1 / 32)) < 1e-6
    weights = tiny_weights()
    state = {name: torch.from_numpy(array) for name, array in weights.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not unexpected, unexpected
    assert all("layers" not in name for name in missing), missing  # embedding, final norm, head

    rng = np.random.default_rng(181)
    d = TINY["d_model"]
    x = rng.standard_normal((2, 6, d)).astype(np.float32)
    spread = np.array([0, 1, 2, 3, 40, 1000])
    out = {
        "checksum": np.array(checksum(weights)),
        "x": x,
        "spread": spread,
    }
    for layer in range(TINY["n_layers"]):
        out[f"y_{layer}"] = run_layer(model, layer, x, np.arange(6))
    out["y_spread"] = run_layer(model, 0, x, spread)
    out["y_offset"] = run_layer(model, 2, x, 100 + np.arange(6))
    np.savez_compressed(HERE / "tiny.npz", **out)
    print(f"wrote tiny.npz ({(HERE / 'tiny.npz').stat().st_size} bytes)")


def granite() -> None:
    directory = assets.resolve(SMALL, assets.models_dir(find_root()))
    tokenizer = AutoTokenizer.from_pretrained(directory)
    model = AutoModelForCausalLM.from_pretrained(directory, dtype=torch.float32).eval()
    ids = tokenizer(PROMPT, return_tensors="pt").input_ids
    assert ids.shape[1] == len(GAPPED), ids.shape
    with torch.no_grad():
        states = model(ids, output_hidden_states=True).hidden_states
    states = [state.numpy().astype(np.float32) for state in states]  # (1, T, 1024) each
    out = {
        "revision": np.array(SMALL.revision),
        "prompt": np.array(PROMPT),
        "ids": ids[0].numpy().astype(np.int32),
        "columns": COLUMNS,
        "gapped": GAPPED,
        "x_1": states[1],
        "x_13": states[13],
        "y_0": states[1][..., COLUMNS],
        "y_1": states[2][..., COLUMNS],
        "y_13": states[14][..., COLUMNS],
        "y_13_gapped": run_layer(model, 13, states[13], GAPPED)[..., COLUMNS],
    }
    # The hook must reproduce the ordinary forward pass before it is trusted with gaps.
    again = run_layer(model, 13, states[13], np.arange(ids.shape[1]))
    assert np.abs(again - states[14]).max() < 1e-4
    np.savez_compressed(HERE / "granite-4.0-350m.npz", **out)
    size = (HERE / "granite-4.0-350m.npz").stat().st_size
    scale = {name: float(np.abs(out[name]).max()) for name in ("x_1", "x_13", "y_13")}
    print(f"wrote granite-4.0-350m.npz ({size} bytes); {ids.shape[1]} tokens; max |h|: {scale}")


if __name__ == "__main__":
    tiny()
    granite()
