"""Regenerate the fixture of per-layer-embeddings.

    uv run --project python --extra jax --extra mlx --extra models \
        python problems/block-components/per-layer-embeddings/fixtures/make.py

Needs the ``models`` extra (``transformers``); no weights are downloaded. Every expected
array is computed by ``Gemma4TextModel`` and ``Gemma4TextDecoderLayer`` of
``models/gemma4/modeling_gemma4.py``, so the tests compare a solution with the shipped
code and not only with this problem's own reference.

A ``Gemma4TextModel`` is built with a tiny configuration (hidden 16, 3 layers,
``hidden_size_per_layer_input`` 4, vocabulary 11; ``rms_norm_eps`` 1e-6 and
``hidden_activation`` "gelu_pytorch_tanh" are the defaults of the class and the values
in the ``config.json`` of ``google/gemma-4-E2B-it``, read on 2026-11-21) and random
weights of order 1.

- ``tokens``, ``h0``, ``table``, ``w_model``, ``norm_gain`` -> ``ple``:
  ``get_per_layer_inputs(tokens, None)`` then ``project_per_layer_inputs(h0, .)``, with
  ``h0 = embed_tokens(tokens)`` (the scaled embeddings the model itself passes).
  ``h0_small`` -> ``ple_small``: the same call with ``h0 / 1000``, where the epsilon of
  the norm is as large as the mean square it is added to.
- ``inject_h``, ``inject_p`` -> ``inject_y``: ``Gemma4TextDecoderLayer.forward`` of layer
  0 with attention and MLP replaced by modules that return zeros, so both residual adds
  leave the stream unchanged and the layer's output is the injection alone.
- ``layer{i}.<name>``, ``layer_x`` -> ``layer{i}_y``: the same ``forward`` of layer ``i``
  with the real ``Gemma4TextMLP`` and with ``self_attn`` replaced by single-head softmax
  attention under a causal mask (the stand-in this problem uses for Gemma 4's
  attention), given ``ple`` row slice ``[:, i]`` as the model's loop does.
  ``layer_scalar`` is a buffer of ones in a new module; it is set to a value other than
  1 here so that it is visible.

The script also asserts where the injection sits: with the MLP replaced by a constant,
the layer's output equals the injection applied to the stream after the MLP's add.
"""

from pathlib import Path

import numpy as np
import torch
import transformers
from torch import nn
from transformers.models.gemma4 import modeling_gemma4 as gemma4
from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig

HERE = Path(__file__).parent
HIDDEN, LAYERS, D_PLE, VOCAB, INTER, STEPS = 16, 3, 4, 11, 24, 5


class StandInAttention(nn.Module):
    """Single-head softmax attention under a causal mask, in place of Gemma4TextAttention."""

    def __init__(self, hidden: int):
        super().__init__()
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.k_proj = nn.Linear(hidden, hidden, bias=False)
        self.v_proj = nn.Linear(hidden, hidden, bias=False)
        self.o_proj = nn.Linear(hidden, hidden, bias=False)

    def forward(self, hidden_states, **kwargs):
        q, k, v = (proj(hidden_states) for proj in (self.q_proj, self.k_proj, self.v_proj))
        scores = q @ k.transpose(-1, -2) / q.shape[-1] ** 0.5
        steps = scores.shape[-1]
        causal = torch.tril(torch.ones(steps, steps, dtype=torch.bool))
        weights = torch.softmax(scores.masked_fill(~causal, float("-inf")), dim=-1)
        return self.o_proj(weights @ v), None


class Constant(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, hidden_states, **kwargs):
        return self.value.expand_as(hidden_states)


class ConstantAttention(Constant):
    def forward(self, hidden_states, **kwargs):
        return super().forward(hidden_states), None


def randomize(module: nn.Module) -> None:
    """Weights of order 1 / sqrt(fan_in) and norm gains in [0.5, 1.5]."""
    with torch.no_grad():
        for name, p in module.named_parameters():
            if p.ndim == 1:
                p.uniform_(0.5, 1.5)
            elif "embed_tokens" in name:
                p.normal_(0.0, 1.0)
            else:
                p.normal_(0.0, p.shape[1] ** -0.5)


def run_layer(layer, x, per_layer_input):
    with torch.no_grad():
        return layer(x[None].clone(), per_layer_input[None])[0]


def main() -> None:
    torch.manual_seed(2304)
    config = Gemma4TextConfig(
        vocab_size=VOCAB,
        vocab_size_per_layer_input=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=INTER,
        num_hidden_layers=LAYERS,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=HIDDEN,
        hidden_size_per_layer_input=D_PLE,
        layer_types=["full_attention"] * LAYERS,
    )
    assert config.rms_norm_eps == 1e-6
    assert config.hidden_activation == "gelu_pytorch_tanh"
    model = gemma4.Gemma4TextModel(config).eval()
    for layer in model.layers:
        layer.self_attn = StandInAttention(HIDDEN)
    randomize(model)
    assert model.per_layer_input_scale == 2.0**-0.5
    assert model.per_layer_model_projection_scale == HIDDEN**-0.5
    assert float(model.embed_tokens_per_layer.embed_scale) == D_PLE**0.5

    out = {"transformers_version": np.array(transformers.__version__)}
    tokens = torch.tensor([[3, 10, 0, 3, 7]])
    with torch.no_grad():
        h0 = model.embed_tokens(tokens)  # scaled by sqrt(hidden), as the model passes it
        token_part = model.get_per_layer_inputs(tokens, None)
        ple = model.project_per_layer_inputs(h0, token_part)
        ple_small = model.project_per_layer_inputs(h0 / 1000.0, token_part)
    out["tokens"] = tokens[0].numpy()
    out["h0"] = h0[0].numpy()
    out["h0_small"] = (h0 / 1000.0)[0].numpy()
    out["table"] = model.embed_tokens_per_layer.weight.detach().numpy()
    out["w_model"] = model.per_layer_model_projection.weight.detach().numpy()
    out["norm_gain"] = model.per_layer_projection_norm.weight.detach().numpy()
    out["ple"] = ple[0].numpy()
    out["ple_small"] = ple_small[0].numpy()

    # The injection alone: both sublayers return zeros.
    layer = model.layers[0]
    real_attn, real_mlp = layer.self_attn, layer.mlp
    zeros = torch.zeros(HIDDEN)
    layer.self_attn, layer.mlp = ConstantAttention(zeros), Constant(zeros)
    inject_h = torch.randn(STEPS, HIDDEN)
    inject_p = ple[0, :, 0]
    inject_y = run_layer(layer, inject_h, inject_p)
    out["inject_h"], out["inject_p"], out["inject_y"] = (
        inject_h.numpy(),
        inject_p.numpy(),
        inject_y.numpy(),
    )

    # Where the injection sits: after the MLP's residual add.
    delta = torch.randn(HIDDEN)
    layer.mlp = Constant(delta)
    with torch.no_grad():
        after_mlp = inject_h + layer.post_feedforward_layernorm(delta.expand(STEPS, HIDDEN))
    layer.self_attn, layer.mlp = ConstantAttention(zeros), Constant(zeros)
    want = run_layer(layer, after_mlp, inject_p)
    layer.mlp = Constant(delta)
    got = run_layer(layer, inject_h, inject_p)
    assert torch.allclose(got, want, atol=1e-6), "the injection does not follow the MLP's add"
    layer.self_attn, layer.mlp = real_attn, real_mlp

    # Whole layers, with the stand-in attention and the real MLP.
    layer_x = torch.randn(STEPS, HIDDEN)
    out["layer_x"] = layer_x.numpy()
    names = (
        "input_layernorm.weight",
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "post_attention_layernorm.weight",
        "pre_feedforward_layernorm.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
        "post_feedforward_layernorm.weight",
        "per_layer_input_gate.weight",
        "per_layer_projection.weight",
        "post_per_layer_input_norm.weight",
    )
    for i, layer in enumerate(model.layers):
        with torch.no_grad():
            layer.layer_scalar.fill_(0.7 + 0.2 * i)
        state = layer.state_dict()
        for name in names:
            out[f"layer{i}.{name}"] = state[name].numpy()
        out[f"layer{i}.layer_scalar"] = state["layer_scalar"].numpy()
        out[f"layer{i}_y"] = run_layer(layer, layer_x, ple[0, :, i]).numpy()

    np.savez(HERE / "shipped.npz", **out)
    print(f"wrote shipped.npz ({len(out)} arrays, transformers {transformers.__version__})")


if __name__ == "__main__":
    main()
