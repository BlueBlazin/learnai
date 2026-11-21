"""Regenerate the fixture of hyper-connections.

    uv run --project python --extra jax --extra mlx --extra models \
        python problems/block-components/hyper-connections/fixtures/make.py

Needs the ``models`` extra (``transformers``); no weights are downloaded. Every expected
array is computed by ``DeepseekV4HyperConnection`` and ``DeepseekV4DecoderLayer`` of
``models/deepseek_v4/modeling_deepseek_v4.py``, so the tests compare a solution with the
shipped code and not only with this problem's own reference.

The configuration is tiny (hidden 8) with the stream settings of
``deepseek-ai/DeepSeek-V4-Pro-0813`` (``config.json`` read on 2026-11-21: ``hc_mult`` 4,
``hc_sinkhorn_iters`` 20, ``hc_eps`` 1e-6, ``rms_norm_eps`` 1e-6) unless a run says
otherwise. The module's parameters are created uninitialized and the model's
initializer sets ``base`` to 0 and ``scale`` to 1; here ``fn`` is drawn so that the raw
weights are of order 2, ``base`` of order 1, and ``scale`` from [0.5, 1.5], so that
each of them shows in the result.

- ``w{iters}_*`` for ``iters`` in 1, 2, 20: one ``DeepseekV4HyperConnection.forward``
  on the same streams and parameters: ``streams``, ``fn``, ``base``, ``scale`` ->
  ``collapsed``, ``post``, ``comb``.
- ``eps_*``: the same with ``hc_eps = 0.01`` and 2 iterations, streams of order 1 and
  (``eps_small_*``) of order 1e-3. ``rms_norm_eps`` stays 1e-6.
- ``n3_*``: 3 streams, 20 iterations.
- ``layer_*``: ``DeepseekV4DecoderLayer.forward`` with attention and MLP replaced by
  modules that return fixed tensors ``y_attn`` and ``y_mlp`` whatever they are given.
  The output is then ``hc_merge(s1, y_mlp, *weights(s1, ffn_hc))`` with
  ``s1 = hc_merge(streams, y_attn, *weights(streams, attn_hc))``; the fixture stores the
  parameters of both hyper-connections, the two sublayer outputs, and the layer's output.
"""

from pathlib import Path

import numpy as np
import torch
import transformers
from torch import nn
from transformers.models.deepseek_v4 import modeling_deepseek_v4 as dsv4
from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config

HERE = Path(__file__).parent
HIDDEN, STEPS = 8, 5


def config(hc_mult: int = 4, iters: int = 20, eps: float = 1e-6) -> DeepseekV4Config:
    cfg = DeepseekV4Config(
        vocab_size=32,
        hidden_size=HIDDEN,
        head_dim=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        moe_intermediate_size=8,
        n_routed_experts=2,
        num_experts_per_tok=1,
        q_lora_rank=4,
        o_lora_rank=4,
        o_groups=2,
        index_n_heads=2,
        index_head_dim=4,
        index_topk=4,
        max_position_embeddings=64,
        hc_mult=hc_mult,
        hc_sinkhorn_iters=iters,
        hc_eps=eps,
    )
    assert cfg.rms_norm_eps == 1e-6
    return cfg


def randomize(hc: dsv4.DeepseekV4HyperConnection) -> None:
    with torch.no_grad():
        hc.fn.normal_(0.0, 2.0 * hc.fn.shape[1] ** -0.5)
        hc.base.normal_(0.0, 1.0)
        hc.scale.uniform_(0.5, 1.5)


def run(hc, streams, prefix, out):
    with torch.no_grad():
        post, comb, collapsed = hc(streams[None])
    out[f"{prefix}_streams"] = streams.numpy()
    out[f"{prefix}_fn"] = hc.fn.detach().numpy().copy()
    out[f"{prefix}_base"] = hc.base.detach().numpy().copy()
    out[f"{prefix}_scale"] = hc.scale.detach().numpy().copy()
    out[f"{prefix}_collapsed"] = collapsed[0].numpy()
    out[f"{prefix}_post"] = post[0].numpy()
    out[f"{prefix}_comb"] = comb[0].numpy()


class Fixed(nn.Module):
    """A sublayer that ignores its input. `pair` mimics attention's (output, weights)."""

    def __init__(self, value: torch.Tensor, pair: bool):
        super().__init__()
        self.value, self.pair = value, pair

    def forward(self, hidden_states, **kwargs):
        return (self.value, None) if self.pair else self.value


def main() -> None:
    torch.manual_seed(2305)
    out = {"transformers_version": np.array(transformers.__version__)}

    assert DeepseekV4Config().hc_mult == 4
    assert DeepseekV4Config().hc_sinkhorn_iters == 20
    assert DeepseekV4Config().hc_eps == 1e-6

    # The same streams and parameters at 1, 2, and 20 iterations.
    streams = torch.randn(STEPS, 4, HIDDEN)
    first = dsv4.DeepseekV4HyperConnection(config(iters=1))
    randomize(first)
    for iters in (1, 2, 20):
        hc = dsv4.DeepseekV4HyperConnection(config(iters=iters))
        hc.load_state_dict(first.state_dict())
        run(hc, streams, f"w{iters}", out)

    # A large hc_eps, where each of its four uses shows; and small streams, where the
    # epsilon of the norm (still 1e-6) shows.
    hc = dsv4.DeepseekV4HyperConnection(config(iters=2, eps=0.01))
    randomize(hc)
    run(hc, torch.randn(STEPS, 4, HIDDEN), "eps", out)
    run(hc, torch.randn(STEPS, 4, HIDDEN) * 1e-3, "eps_small", out)

    hc = dsv4.DeepseekV4HyperConnection(config(hc_mult=3))
    randomize(hc)
    run(hc, torch.randn(STEPS, 3, HIDDEN), "n3", out)

    # A whole decoder layer with fixed sublayer outputs.
    layer = dsv4.DeepseekV4DecoderLayer(config(), 0).eval()
    randomize(layer.attn_hc)
    randomize(layer.ffn_hc)
    y_attn, y_mlp = torch.randn(1, STEPS, HIDDEN), torch.randn(1, STEPS, HIDDEN)
    layer.self_attn, layer.mlp = Fixed(y_attn, pair=True), Fixed(y_mlp, pair=False)
    streams = torch.randn(STEPS, 4, HIDDEN)
    with torch.no_grad():
        result = layer(streams[None].clone())
    out["layer_streams"] = streams.numpy()
    out["layer_y_attn"], out["layer_y_mlp"] = y_attn[0].numpy(), y_mlp[0].numpy()
    for name, hc in (("attn", layer.attn_hc), ("ffn", layer.ffn_hc)):
        out[f"layer_{name}_fn"] = hc.fn.detach().numpy()
        out[f"layer_{name}_base"] = hc.base.detach().numpy()
        out[f"layer_{name}_scale"] = hc.scale.detach().numpy()
    out["layer_out"] = result[0].numpy()

    np.savez(HERE / "shipped.npz", **out)
    print(f"wrote shipped.npz ({len(out)} arrays, transformers {transformers.__version__})")


if __name__ == "__main__":
    main()
