"""Regenerate the fixture of shared-index-layers.

    uv run --project python --extra jax --extra mlx --extra models \
        python problems/sparse-attention/shared-index-layers/fixtures/make.py

Needs the ``models`` extra (``transformers``); no weights are downloaded. The fixture
records how ``GlmMoeDsaModel`` of ``models/glm_moe_dsa/modeling_glm_moe_dsa.py``
executes cross-layer index sharing, so the tests compare a solution's sharing pattern
with the shipped model's and not only with the kit's oracle.

A model with random weights and small sizes is built with the ``indexer_types`` of the
first 11 layers of ``zai-org/GLM-5.3`` (``config.json`` read on 2026-11-21: 78 layers,
three "full", then "shared", "shared", "shared", "full" repeating; the list below is
copied from the file). Its forward pass is run on 10 tokens with every layer's
attention and indexer wrapped. Saved in ``shipped.json``:

- ``types``: the 11 layer types;
- ``source``: for each layer, the layer whose indexer produced the indices that the
  layer's attention used (itself for a full layer);
- ``indexer_calls``: how many indexers ran;
- ``has_indexer``: per layer, whether the attention module owns an indexer at all;
- ``rule_types``: what ``GlmMoeDsaConfig`` derives for 11 layers from GLM-5.3's
  ``index_topk_freq`` 4 and ``index_skip_topk_offset`` 3 when no list is given;
- ``error``: the exception type and message of a model whose first layer is shared.
"""

import json
from pathlib import Path

import torch
import transformers
from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import GlmMoeDsaConfig
from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaModel

HERE = Path(__file__).parent
# zai-org/GLM-5.3 config.json, indexer_types[:11].
TYPES = ["full"] * 3 + ["shared"] * 3 + ["full"] + ["shared"] * 3 + ["full"]
SMALL = {
    "vocab_size": 32,
    "hidden_size": 32,
    "intermediate_size": 16,
    "moe_intermediate_size": 16,
    "num_attention_heads": 2,
    "num_key_value_heads": 2,
    "n_routed_experts": 2,
    "num_experts_per_tok": 1,
    "kv_lora_rank": 8,
    "q_lora_rank": 16,
    "qk_rope_head_dim": 4,
    "qk_nope_head_dim": 4,
    "v_head_dim": 8,
    "index_topk": 3,
    "index_head_dim": 8,
    "index_n_heads": 4,
    "max_position_embeddings": 64,
    "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0},
}


def run(types: list[str]) -> dict:
    torch.manual_seed(2204)
    config = GlmMoeDsaConfig(num_hidden_layers=len(types), indexer_types=list(types), **SMALL)
    model = GlmMoeDsaModel(config).eval()
    produced: dict[int, torch.Tensor] = {}  # layer -> the indices its indexer returned
    used: dict[int, torch.Tensor] = {}  # layer -> the indices its attention returned (and used)

    for i, layer in enumerate(model.layers):
        attention = layer.self_attn
        if attention.indexer is not None:
            real_indexer = attention.indexer.forward

            def indexer(*args, _real=real_indexer, _i=i, **kwargs):
                produced[_i] = _real(*args, **kwargs)
                return produced[_i]

            attention.indexer.forward = indexer
        real_attention = attention.forward

        def attend(*args, _real=real_attention, _i=i, **kwargs):
            out = _real(*args, **kwargs)
            used[_i] = out[2]
            return out

        attention.forward = attend

    with torch.no_grad():
        model(input_ids=torch.randint(0, 32, (1, 10)))
    source = []
    for i in range(len(types)):
        owners = [j for j, indices in produced.items() if indices is used[i]]
        assert len(owners) == 1, (i, owners)
        source.append(owners[0])
    return {
        "source": source,
        "indexer_calls": len(produced),
        "has_indexer": [layer.self_attn.indexer is not None for layer in model.layers],
    }


def main() -> None:
    out = {"transformers_version": transformers.__version__, "types": TYPES, **run(TYPES)}
    rule = GlmMoeDsaConfig(
        num_hidden_layers=len(TYPES), index_topk_freq=4, index_skip_topk_offset=3, **SMALL
    )
    out["rule_types"] = list(rule.indexer_types)
    try:
        run(["shared", "full", "shared"])
    except ValueError as exc:  # any other exception is a finding: let it propagate
        out["error"] = {"type": type(exc).__name__, "message": str(exc)}
    (HERE / "shipped.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
