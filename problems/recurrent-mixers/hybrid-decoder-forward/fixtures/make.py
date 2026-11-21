"""Regenerate the fixture of hybrid-decoder-forward.

    uv run --project python --extra jax --extra mlx --extra models \
        python problems/recurrent-mixers/hybrid-decoder-forward/fixtures/make.py

Needs the ``models`` extra (``transformers``); no weights are downloaded. Every expected
array is computed by a ``Qwen3_5ForCausalLM`` of ``transformers``
(``models/qwen3_5/modeling_qwen3_5.py``) built from a tiny ``Qwen3_5TextConfig`` and
holding the kit's seeded weights ``hybrid_params(cfg, SEED)``, so the tests compare a
solution with the shipped code and not only with this problem's reference.

Per configuration ``{name}`` (``CONFIGS`` below; the same objects are built in
``tests.py``) and the 8 tokens ``{name}_tokens``:

- ``{name}_one_pass``: logits of one call with all tokens and no cache. The library runs
  its chunked delta rule in every linear-attention layer (asserted).
- ``{name}_cached_{i}``: logits of the same tokens fed with a ``DynamicCache`` in the
  pieces ``CHUNKINGS[i]``. A one-token call that continues a cache runs the library's
  per-token delta rule, every other call its chunked rule (both asserted to have run).
- ``{name}_state_{layer}_{0,1}``: the cache after the last chunking, in this problem's
  layout. The library differs in three ways, all undone here: the kit's norm gains are
  stored minus 1 in the module (its block norms, final norm, and QK norms compute
  ``x_normed * (1 + weight)``); its delta memory is ``(Hv, Dk, Dv)`` and is stored here
  transposed; it keeps ``K`` rows of convolution state as ``(C, K)``, of which the last
  ``K - 1`` are stored here as ``(K - 1, C)``.

``schedule_{n}_{interval}`` holds the ``layer_types`` that ``Qwen3_5TextConfig`` derives
from ``full_attention_interval``, as an array of strings.

Sizes: the kit's small defaults. ``rope_theta`` 1e7, ``partial_rotary_factor`` 0.25,
kernel 4, ``rms_norm_eps`` 1e-6, an interval of 4, and tied embeddings are the values in
the ``config.json`` of ``Qwen/Qwen3.5-0.8B`` (read on 2026-11-21).
"""

from pathlib import Path
from unittest import mock

import numpy as np
import torch
import transformers
from transformers import DynamicCache
from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

from learnai.kit import HybridConfig, hybrid_params

HERE = Path(__file__).parent
SEED, STEPS = 11, 8
FULL, LINEAR = "full_attention", "linear_attention"
CONFIGS = {
    "base": HybridConfig(rope_base=1e7),  # linear, linear, linear, full
    "deep": HybridConfig(n_layers=8, rope_base=1e7),  # attention at layers 3 and 7
    "mixed": HybridConfig(
        n_layers=5, rope_base=1e7, layer_types=(FULL, LINEAR, FULL, LINEAR, LINEAR)
    ),
}
CHUNKINGS = ([6, 1, 1], [1, 1, 1, 5])
SCHEDULES = ((24, 4), (4, 4), (7, 3), (5, 1), (4, 6), (9, 2))
ZERO_CENTERED = (
    "input_layernorm.weight", "post_attention_layernorm.weight", "q_norm.weight",
    "k_norm.weight", "model.norm.weight",
)  # fmt: skip


def text_config(cfg: HybridConfig) -> Qwen3_5TextConfig:
    return Qwen3_5TextConfig(
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


def tiny_qwen(cfg: HybridConfig, params: dict[str, np.ndarray]) -> qwen.Qwen3_5ForCausalLM:
    model = qwen.Qwen3_5ForCausalLM(text_config(cfg)).eval().float()
    expected = model.state_dict()
    state = {}
    for name, value in params.items():
        tensor = torch.from_numpy(value.copy())
        state[name] = tensor - 1.0 if name.endswith(ZERO_CENTERED) else tensor
        assert expected[name].shape == state[name].shape, name
    state["lm_head.weight"] = torch.from_numpy(params["model.embed_tokens.weight"].copy())
    assert set(expected) == set(state)  # the kit's names are the checkpoint's names
    model.load_state_dict(state)
    return model


def main() -> None:
    out = {"transformers_version": np.array(transformers.__version__), "seed": np.array(SEED)}
    rng = np.random.default_rng(SEED)
    for name, cfg in CONFIGS.items():
        model = tiny_qwen(cfg, hybrid_params(cfg, SEED))
        tokens = rng.integers(0, cfg.vocab_size, STEPS)
        ids = torch.from_numpy(tokens)[None]
        linear = cfg.layers.count(LINEAR)
        chunk = mock.patch.object(
            qwen, "torch_chunk_gated_delta_rule", wraps=qwen.torch_chunk_gated_delta_rule
        )
        step = mock.patch.object(
            qwen, "torch_recurrent_gated_delta_rule", wraps=qwen.torch_recurrent_gated_delta_rule
        )
        out[f"{name}_tokens"] = tokens
        with chunk as chunked, step as stepped, torch.no_grad():
            out[f"{name}_one_pass"] = model(ids).logits[0].numpy()
            assert (chunked.call_count, stepped.call_count) == (linear, 0)
            for index, sizes in enumerate(CHUNKINGS):
                chunked.reset_mock()
                stepped.reset_mock()
                cache, pieces, start = DynamicCache(config=model.config), [], 0
                for size in sizes:
                    result = model(ids[:, start : start + size], past_key_values=cache)
                    cache = result.past_key_values
                    pieces.append(result.logits[0].numpy())
                    start += size
                # The per-token rule needs a state: the first call always runs the chunked one.
                ones = sum(size == 1 for size in sizes[1:])
                assert stepped.call_count == linear * ones, (sizes, stepped.call_count)
                assert chunked.call_count == linear * (len(sizes) - ones)
                out[f"{name}_cached_{index}"] = np.concatenate(pieces)
        for layer, kind in enumerate(cfg.layers):
            theirs = cache.layers[layer]
            if kind == FULL:
                pair = (theirs.keys[0], theirs.values[0])
            else:
                assert theirs.conv_states[0].shape == (1, cfg.gdn.conv_dim, cfg.conv_kernel)
                pair = (
                    theirs.conv_states[0][0].T[1:],
                    theirs.recurrent_states[0][0].transpose(-1, -2),
                )
            for part, value in enumerate(pair):
                out[f"{name}_state_{layer}_{part}"] = value.contiguous().numpy()
        worst = max(
            np.abs(out[f"{name}_cached_{i}"] - out[f"{name}_one_pass"]).max()
            for i in range(len(CHUNKINGS))
        )
        print(
            f"{name}: layers {cfg.layers}, largest |logit| "
            f"{np.abs(out[f'{name}_one_pass']).max():.2f}, cached versus one pass {worst:.1e}"
        )
    for n_layers, interval in SCHEDULES:
        theirs = Qwen3_5TextConfig(num_hidden_layers=n_layers, full_attention_interval=interval)
        out[f"schedule_{n_layers}_{interval}"] = np.array(theirs.layer_types)
    np.savez_compressed(HERE / "shipped.npz", **out)
    print(f"wrote shipped.npz ({len(out)} arrays, transformers {transformers.__version__})")


if __name__ == "__main__":
    main()
