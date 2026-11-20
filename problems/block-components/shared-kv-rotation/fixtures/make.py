"""Regenerate the fixture of shared-kv-rotation.

    uv run --project python --extra jax --extra mlx --extra models \
        python problems/block-components/shared-kv-rotation/fixtures/make.py

Needs the ``models`` extra (``transformers``); no weights are downloaded. The expected
outputs come from the attention code of
``transformers/models/deepseek_v4/modeling_deepseek_v4.py``, run on random queries and
one random key-value tensor with the sizes of ``deepseek-ai/DeepSeek-V4-Pro-0813``
(``config.json`` read on 2026-11-21): head width 512, of which the trailing 64 channels
(``qk_rope_head_dim``) rotate, one key-value head, ``rope_theta`` 10000.

The lines below are the core of ``DeepseekV4Attention.forward`` with the library's own
functions: ``DeepseekV4RotaryEmbedding`` (the "main" tables), ``apply_rotary_pos_emb``
on the queries and on the shared tensor, ``eager_attention_forward`` with that tensor
as key and as value, and ``apply_rotary_pos_emb`` with ``-sin`` on the output. What the
module does around them is left out: the projections and norms that produce ``q`` and
``kv``, the compressed entries, and the output projection. The per-head attention sinks
are set to ``-inf``, which gives them probability zero.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import transformers
from transformers.models.deepseek_v4 import modeling_deepseek_v4 as v4
from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config

HERE = Path(__file__).parent
HEADS, TOKENS = 2, 5


def attend(config, rotary, q, kv, positions):
    """q: (1, H, T, d), kv: (1, 1, T, d), positions: (T,). Causal self-attention."""
    cos, sin = rotary(q, position_ids=positions[None], layer_type="main")
    q = v4.apply_rotary_pos_emb(q, cos, sin)
    kv = v4.apply_rotary_pos_emb(kv, cos, sin)
    causal = torch.full((TOKENS, TOKENS), float("-inf")).triu(1)[None, None]
    module = SimpleNamespace(
        sinks=torch.full((HEADS,), float("-inf")), num_key_value_groups=HEADS, training=False
    )
    out, _ = v4.eager_attention_forward(module, q, kv, kv, causal, scaling=config.head_dim**-0.5)
    # The module's line, verbatim in effect: its result is laid out (B, T, H, d).
    out = v4.apply_rotary_pos_emb(out.transpose(1, 2), cos, -sin).transpose(1, 2)
    return out.transpose(1, 2)  # (B, H, T, d), the layout of this problem


def main() -> None:
    config = DeepseekV4Config(head_dim=512, qk_rope_head_dim=64, rope_theta=10000)
    rotary = v4.DeepseekV4RotaryEmbedding(config)
    rng = np.random.default_rng(2303)
    q = torch.from_numpy(rng.standard_normal((1, HEADS, TOKENS, 512)).astype(np.float32))
    kv = torch.from_numpy(rng.standard_normal((1, 1, TOKENS, 512)).astype(np.float32))
    # Rows of unit RMS, as the model's norms produce, times a gain that makes the
    # attention weights uneven.
    q = 3.0 * q / q.pow(2).mean(-1, keepdim=True).sqrt()
    kv = kv / kv.pow(2).mean(-1, keepdim=True).sqrt()
    out = {
        "transformers_version": np.array(transformers.__version__),
        "rope_dim": np.array(config.qk_rope_head_dim),
        "base": np.array(float(config.rope_parameters["main"]["rope_theta"])),
        "q": q[0].numpy(),
        "kv": kv[0, 0].numpy(),
    }
    for name, start in (("near", 0), ("far", 1500)):
        positions = start + torch.arange(TOKENS)
        out[f"pos_{name}"] = positions.numpy()
        out[f"out_{name}"] = attend(config, rotary, q, kv, positions)[0].numpy()
    np.savez_compressed(HERE / "deepseek_v4.npz", **out)
    for key, value in out.items():
        print(key, value.shape, value.dtype)


if __name__ == "__main__":
    main()
