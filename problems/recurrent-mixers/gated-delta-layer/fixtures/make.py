"""Regenerate the fixture of gated-delta-layer.

    uv run --project python --extra jax --extra mlx --extra models \
        python problems/recurrent-mixers/gated-delta-layer/fixtures/make.py

Needs the ``models`` extra (``transformers``); no weights are downloaded. The expected
arrays are what ``Qwen3_5GatedDeltaNet`` of ``transformers``
(``models/qwen3_5/modeling_qwen3_5.py``) returns when it holds the kit's seeded weights
``gdn_params(GDNConfig(), seed=SEED)``: 2 key heads and 4 value heads of width 8,
kernel 4, model width 32, ``rms_norm_eps`` 1e-6 (the value in the ``config.json`` of
``Qwen/Qwen3.5-0.8B``, read on 2026-11-21).

- ``out_one_pass``: one call with all 12 tokens and no cache. The library runs its
  chunked delta rule (asserted below).
- ``out_cached``, ``S``, ``conv_state``: the same tokens with a ``DynamicCache``, 5 in
  the first call and then one per call. The library runs its per-token rule for the
  last 7 (asserted below).

Two differences of layout are undone so that the fixture is in this problem's layout:
the library's memory is ``(Hv, Dk, Dv)`` and is stored here transposed, ``(Hv, Dv,
Dk)``; the library keeps ``K`` rows of convolution state as ``(C, K)`` and the fixture
keeps the last ``K - 1`` of them as ``(K - 1, C)``.
"""

from pathlib import Path
from unittest import mock

import numpy as np
import torch
import transformers
from transformers import DynamicCache
from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

from learnai.kit import GDNConfig, gdn_params

HERE = Path(__file__).parent
SEED, STEPS, PREFILL = 7, 12, 5


def main() -> None:
    cfg = GDNConfig()
    params = gdn_params(cfg, SEED)
    config = Qwen3_5TextConfig(
        hidden_size=cfg.d_model,
        num_hidden_layers=1,
        layer_types=["linear_attention"],
        rms_norm_eps=cfg.norm_eps,
        linear_conv_kernel_dim=cfg.conv_kernel,
        linear_key_head_dim=cfg.key_head_dim,
        linear_value_head_dim=cfg.value_head_dim,
        linear_num_key_heads=cfg.n_key_heads,
        linear_num_value_heads=cfg.n_value_heads,
    )
    layer = qwen.Qwen3_5GatedDeltaNet(config, 0).eval().float()
    layer.load_state_dict({name: torch.from_numpy(value.copy()) for name, value in params.items()})

    rng = np.random.default_rng(SEED)
    x = rng.standard_normal((STEPS, cfg.d_model)).astype(np.float32)
    tokens = torch.from_numpy(x)[None]
    chunk = mock.patch.object(
        qwen, "torch_chunk_gated_delta_rule", wraps=qwen.torch_chunk_gated_delta_rule
    )
    step = mock.patch.object(
        qwen, "torch_recurrent_gated_delta_rule", wraps=qwen.torch_recurrent_gated_delta_rule
    )
    with chunk as chunked, step as stepped, torch.no_grad():
        one_pass = layer(tokens)[0].numpy()
        assert (chunked.call_count, stepped.call_count) == (1, 0)
        cache = DynamicCache(config=config)
        pieces = [layer(tokens[:, :PREFILL], cache_params=cache)]
        pieces += [layer(tokens[:, t : t + 1], cache_params=cache) for t in range(PREFILL, STEPS)]
        assert (chunked.call_count, stepped.call_count) == (2, STEPS - PREFILL)
    theirs = cache.layers[0]
    assert theirs.conv_states[0].shape == (1, cfg.conv_dim, cfg.conv_kernel)
    out = {
        "transformers_version": np.array(transformers.__version__),
        "seed": np.array(SEED),
        "x": x,
        "out_one_pass": one_pass,
        "out_cached": torch.cat(pieces, dim=1)[0].numpy(),
        "S": theirs.recurrent_states[0][0].transpose(-1, -2).contiguous().numpy(),
        "conv_state": theirs.conv_states[0][0].T[1:].contiguous().numpy(),
    }
    np.savez_compressed(HERE / "shipped.npz", **out)
    for key, value in out.items():
        print(key, value.shape, value.dtype)
    print("one pass versus cached:", np.abs(out["out_one_pass"] - out["out_cached"]).max())


if __name__ == "__main__":
    main()
