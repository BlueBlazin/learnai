"""Regenerate the fixture of yarn-frequencies.

    uv run --project python --extra jax --extra mlx --extra models \
        python problems/block-components/yarn-frequencies/fixtures/make.py

Needs the ``models`` extra (``transformers``); no weights are downloaded. Everything
stored is computed by ``_compute_yarn_parameters`` of
``transformers/modeling_rope_utils.py``, reached through each model's own rotary
module and configuration class, for the YaRN parameters in two ``config.json`` files
read on 2026-11-21:

- ``gpt_*``: ``openai/gpt-oss-20b``. Head width 64, ``rope_theta`` 150000, factor 32,
  ``original_max_position_embeddings`` 4096, ``beta_fast`` 32, ``beta_slow`` 1,
  ``truncate`` false. Also the cos and sin tables of ``GptOssRotaryEmbedding`` at a
  few positions, which carry the factor ``0.1 * ln(32) + 1``.
- ``ds_*``: ``deepseek-ai/DeepSeek-V4-Pro-0813``, the rotary slice of its compressed
  attention layers. 64 rotated channels (``qk_rope_head_dim``), ``compress_rope_theta``
  160000, factor 16, ``original_max_position_embeddings`` 65536, ``beta_fast`` 32,
  ``beta_slow`` 1, and no ``truncate`` key, so the library truncates. The configuration
  class sets ``attention_factor`` to 1 for this model, so only the frequencies are stored.
- ``clamp_*``: not a model. ``_compute_yarn_parameters`` for width 16, base 100, factor
  8, original length 4194304, ``beta_fast`` 20000, ``beta_slow`` 1, with and without
  ``truncate``: parameters for which the upper clamp of the ramp changes the result.
"""

from pathlib import Path

import numpy as np
import torch
import transformers
from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4RotaryEmbedding
from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssRotaryEmbedding

HERE = Path(__file__).parent
POSITIONS = [0, 1, 2, 4095, 4096, 131071]

GPT_OSS = {
    "rope_type": "yarn",
    "rope_theta": 150000,
    "factor": 32.0,
    "original_max_position_embeddings": 4096,
    "beta_fast": 32.0,
    "beta_slow": 1.0,
    "truncate": False,
}
CLAMPED = {
    "rope_type": "yarn",
    "rope_theta": 100.0,
    "factor": 8.0,
    "original_max_position_embeddings": 4194304,
    "beta_fast": 20000.0,
    "beta_slow": 1.0,
}
DEEPSEEK_V4 = {
    "type": "yarn",
    "factor": 16,
    "original_max_position_embeddings": 65536,
    "beta_fast": 32,
    "beta_slow": 1,
}


def main() -> None:
    out = {"transformers_version": np.array(transformers.__version__)}

    config = GptOssConfig(head_dim=64, rope_parameters=dict(GPT_OSS))
    rotary = GptOssRotaryEmbedding(config)
    out["gpt_inv_freq"] = rotary.inv_freq.numpy()
    out["gpt_table_factor"] = np.array(rotary.attention_scaling)
    cos, sin = rotary(torch.zeros(1, 1, 1, dtype=torch.float32), torch.tensor([POSITIONS]))
    out["gpt_positions"] = np.array(POSITIONS)
    out["gpt_cos"], out["gpt_sin"] = cos[0].numpy(), sin[0].numpy()

    config = DeepseekV4Config(
        head_dim=512,
        qk_rope_head_dim=64,
        rope_theta=10000,
        compress_rope_theta=160000,
        rope_scaling=dict(DEEPSEEK_V4),
    )
    rotary = DeepseekV4RotaryEmbedding(config)
    out["ds_inv_freq"] = rotary.compress_inv_freq.numpy()
    out["ds_table_factor"] = np.array(rotary.compress_attention_scaling)

    # Parameters under which the upper clamp of the ramp, min(high, dim - 1), binds and
    # changes the result: the unclamped bound is 23.3 for a 16-wide embedding. Not a
    # shipped model; the library's function on the gpt-oss configuration class.
    for name, truncate in (("clamp_inv_freq", False), ("clamp_inv_freq_truncated", True)):
        config = GptOssConfig(head_dim=16, rope_parameters={**CLAMPED, "truncate": truncate})
        out[name] = GptOssRotaryEmbedding(config).inv_freq.numpy()

    np.savez_compressed(HERE / "shipped.npz", **out)
    for key, value in out.items():
        print(key, value.shape, value.dtype, value if value.ndim == 0 else "")


if __name__ == "__main__":
    main()
