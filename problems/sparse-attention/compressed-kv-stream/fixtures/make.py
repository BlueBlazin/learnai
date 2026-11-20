"""Regenerate the fixture of compressed-kv-stream.

    uv run --project python --extra jax --extra mlx --extra models \
        python problems/sparse-attention/compressed-kv-stream/fixtures/make.py

Needs the ``models`` extra (``transformers``); no weights are downloaded. The expected
arrays are what ``DeepseekV4HCACompressor`` and ``DeepseekV4HCACache`` of
``models/deepseek_v4/modeling_deepseek_v4.py`` compute, so the tests compare a
solution with the shipped code and not only with this problem's own reference.

Two compressors with random weights are run on a stream that arrives in pieces, each
with its own ``DeepseekV4HCACache``:

- ``w128_*``: the window of ``deepseek-ai/DeepSeek-V4-Pro-0813`` (``config.json`` read on
  2026-11-21: ``compress_ratios`` of 128 on the heavily compressed layers,
  ``rms_norm_eps`` 1e-6), 300 rows in calls of 100, 60, 1, and 139;
- ``w4_*``: a window of 4, 14 rows in calls of 3, 1, 9, and 1.

Both use a head width of 16 (the checkpoint's is 512). Saved per run:

- ``kv``, ``gate``: the compressor's own ``kv_proj`` and ``gate_proj`` outputs for the
  whole stream (the rows the cache buffers);
- ``position_bias``, ``gain`` (the weight of ``kv_norm``), ``rate``, ``cuts``;
- ``entries``: every entry **before** the rotary embedding is applied to it, captured
  as the first argument of the module's ``apply_rotary_pos_emb`` call. The script also
  asserts that the channels the rotation does not touch are equal in the module's
  final output;
- ``buffer_kv``, ``buffer_gate``, ``count``: the cache's buffer and entry count after
  the last call.

``w4_visible`` is the number of entries each of 23 query positions may attend to,
counted from the ``block_bias`` the compressor returns for one stateless call over
positions 0 to 22.
"""

from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
import transformers
from transformers.models.deepseek_v4 import modeling_deepseek_v4 as dsv4
from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config

HERE = Path(__file__).parent
HEAD_DIM = 16


def build(rate: int, seed: int):
    torch.manual_seed(seed)
    config = DeepseekV4Config(
        vocab_size=32,
        hidden_size=24,
        head_dim=HEAD_DIM,
        num_hidden_layers=1,
        num_attention_heads=2,
        layer_types=["heavily_compressed_attention"],
        compress_rates={"compressed_sparse_attention": 4, "heavily_compressed_attention": rate},
        max_position_embeddings=1024,
    )
    assert config.rms_norm_eps == 1e-6
    compressor = dsv4.DeepseekV4HCACompressor(config).eval()
    with torch.no_grad():  # position_bias is created uninitialized; the norm weight is 1
        compressor.position_bias.normal_(0.0, 1.0)
        compressor.kv_norm.weight.uniform_(0.5, 1.5)
        compressor.gate_proj.weight.mul_(4.0)  # gates of order 1, so the softmax is not flat
    return config, compressor


def run(rate: int, steps: int, cuts: list[int], seed: int) -> dict[str, np.ndarray]:
    config, compressor = build(rate, seed)
    cache = dsv4.DeepseekV4HCACache(config)
    past = SimpleNamespace(layers=[cache])
    hidden = torch.randn(1, steps, config.hidden_size)
    before_rope = []
    real_rope = dsv4.apply_rotary_pos_emb

    def spy(x, cos, sin, *args, **kwargs):
        before_rope.append(x.detach().clone()[0, 0])  # (windows of this call, head_dim)
        return real_rope(x, cos, sin, *args, **kwargs)

    start = 0
    with mock.patch.object(dsv4, "apply_rotary_pos_emb", spy), torch.no_grad():
        for stop in cuts:
            positions = torch.arange(start, stop)[None]
            running, _ = compressor(hidden[:, start:stop], None, positions, past, 0)
            start = stop
        kv, gate = compressor.kv_proj(hidden)[0], compressor.gate_proj(hidden)[0]
    entries = torch.cat(before_rope)
    assert cache.entry_count["compressor"] == steps // rate == entries.shape[0]
    # The rotation touches the trailing rope slice only: the other channels of the
    # module's final output are the captured entries.
    rope_dim = int(HEAD_DIM * config.rope_parameters["compress"]["partial_rotary_factor"])
    assert 0 < rope_dim < HEAD_DIM
    assert torch.equal(running[0, 0, :, :-rope_dim], entries[:, :-rope_dim])
    return {
        "kv": kv.numpy(),
        "gate": gate.numpy(),
        "position_bias": compressor.position_bias.detach().numpy(),
        "gain": compressor.kv_norm.weight.detach().numpy(),
        "rate": np.array(rate),
        "cuts": np.array(cuts),
        "entries": entries.numpy(),
        "buffer_kv": cache.buffer_kv["compressor"][0].numpy(),
        "buffer_gate": cache.buffer_gate["compressor"][0].numpy(),
        "count": np.array(cache.entry_count["compressor"]),
    }


def visible_counts(rate: int, steps: int, seed: int) -> np.ndarray:
    """Entries each query position may see, from the block bias of one stateless call."""
    config, compressor = build(rate, seed)
    hidden = torch.randn(1, steps, config.hidden_size)
    with torch.no_grad():
        entries, block_bias = compressor(hidden, None, torch.arange(steps)[None], None, 0)
    assert entries.shape[2] == steps // rate
    return (block_bias[0, 0] == 0).sum(-1).numpy()


def main() -> None:
    out = {"transformers_version": np.array(transformers.__version__)}
    for name, result in (
        ("w128", run(128, 300, [100, 160, 161, 300], seed=2205)),
        ("w4", run(4, 14, [3, 4, 13, 14], seed=2206)),
    ):
        out.update({f"{name}_{key}": value for key, value in result.items()})
    out["w4_visible"] = visible_counts(4, 23, seed=2207)
    np.savez_compressed(HERE / "shipped.npz", **out)
    for key, value in out.items():
        print(key, value.shape, value.dtype)
    print("w4_visible", out["w4_visible"].tolist())
    print(
        "w4 buffer rows", out["w4_buffer_kv"].shape, "w128 buffer rows", out["w128_buffer_kv"].shape
    )


if __name__ == "__main__":
    main()
