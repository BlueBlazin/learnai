"""Regenerate the fixture of partial-rope.

    uv run --project python --extra jax --extra mlx --extra models \
        python problems/block-components/partial-rope/fixtures/make.py

Needs the ``models`` extra (``transformers``); no weights are downloaded. The expected
arrays are what the rotary code shipped in ``transformers`` produces for the RoPE
parameters of two real checkpoints, so the tests compare a solution with the library
and not only with this problem's own reference.

- ``qwen_*``: ``Qwen3_5TextRotaryEmbedding`` and ``apply_rotary_pos_emb`` of
  ``models/qwen3_5/modeling_qwen3_5.py`` with the ``rope_parameters`` of
  ``Qwen/Qwen3.5-0.8B`` (``config.json`` read on 2026-11-21): ``rope_type`` "default",
  ``partial_rotary_factor`` 0.25, ``rope_theta`` 1e7, head width 256.
- ``gemma_*``: ``Gemma4TextRotaryEmbedding`` and ``apply_rotary_pos_emb`` of
  ``models/gemma4/modeling_gemma4.py`` for the full-attention layers of
  ``google/gemma-4-E2B-it`` (same date): ``rope_type`` "proportional",
  ``partial_rotary_factor`` 0.25, ``rope_theta`` 1e6, head width 512. Its inverse
  frequencies come from ``_compute_proportional_rope_parameters``.

``*_near`` uses positions below 64, ``*_far`` positions up to 2047 (compared with the
looser tolerance of the statement).
"""

from pathlib import Path

import numpy as np
import torch
import transformers
from transformers.models.gemma4 import modeling_gemma4 as gemma4
from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen3_5
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

HERE = Path(__file__).parent
NEAR = [0, 1, 2, 17, 63]
FAR = [64, 300, 777, 1500, 2047]

QWEN = {"rope_type": "default", "rope_theta": 10000000, "partial_rotary_factor": 0.25}
GEMMA = {
    "full_attention": {
        "rope_type": "proportional",
        "partial_rotary_factor": 0.25,
        "rope_theta": 1000000.0,
    },
    "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
}


def main() -> None:
    rng = np.random.default_rng(2301)
    out = {"transformers_version": np.array(transformers.__version__)}

    config = Qwen3_5TextConfig(
        hidden_size=1024, num_attention_heads=8, head_dim=256, rope_parameters=dict(QWEN)
    )
    rotary = qwen3_5.Qwen3_5TextRotaryEmbedding(config)
    x = torch.from_numpy(rng.standard_normal((1, 2, 5, 256)).astype(np.float32))
    out["qwen_x"] = x.numpy()
    out["qwen_theta"] = np.array(float(QWEN["rope_theta"]))
    out["qwen_factor"] = np.array(QWEN["partial_rotary_factor"])
    for name, positions in (("near", NEAR), ("far", FAR)):
        cos, sin = rotary(x, torch.tensor([positions]))
        rotated, _ = qwen3_5.apply_rotary_pos_emb(x, x, cos, sin)
        out[f"qwen_pos_{name}"] = np.array(positions)
        out[f"qwen_y_{name}"] = rotated.numpy()

    config = Gemma4TextConfig(rope_parameters={k: dict(v) for k, v in GEMMA.items()})
    head_dim = config.per_layer_config["full_attention"].head_dim
    rotary = gemma4.Gemma4TextRotaryEmbedding(config)
    x = torch.from_numpy(rng.standard_normal((1, 1, 5, head_dim)).astype(np.float32))
    out["gemma_x"] = x.numpy()
    out["gemma_theta"] = np.array(GEMMA["full_attention"]["rope_theta"])
    out["gemma_factor"] = np.array(GEMMA["full_attention"]["partial_rotary_factor"])
    for name, positions in (("near", NEAR), ("far", FAR)):
        cos, sin = rotary(x, torch.tensor([positions]), "full_attention")
        out[f"gemma_pos_{name}"] = np.array(positions)
        out[f"gemma_y_{name}"] = gemma4.apply_rotary_pos_emb(x, cos, sin).numpy()

    np.savez_compressed(HERE / "shipped.npz", **out)
    for key, value in out.items():
        print(key, value.shape, value.dtype)


if __name__ == "__main__":
    main()
