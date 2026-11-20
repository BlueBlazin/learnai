"""Regenerate the fixture of gated-conv-block.

    uv run --project python --extra jax --extra mlx --extra models \
        python problems/recurrent-mixers/gated-conv-block/fixtures/make.py

Needs the ``models`` extra (``transformers``); no weights are downloaded. The expected
arrays are what ``Lfm3ShortConv`` of ``transformers`` (``models/lfm3/modeling_lfm3.py``)
returns for seeded weights with the convolution settings of ``LiquidAI/LFM3-1.7B``
(``config.json`` read on 2026-11-21: ``conv_L_cache`` 7, ``conv_bias`` false), at
width 6 instead of 2048:

- ``y_one_pass``: 12 tokens in one call without a cache.
- ``y_cached``, ``state``: the same tokens with a ``DynamicCache``, 5 in the first call
  and then one per call (the library's one-token update), and the convolution state it
  holds afterwards.

The library keeps the last ``L`` rows of ``u`` as ``(D, L)``; the fixture stores the
last ``L - 1`` of them as ``(L - 1, D)``, this problem's layout. ``w_conv`` is the
module's ``conv.weight`` ``(D, 1, L)`` without its middle axis.
"""

from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import DynamicCache
from transformers.models.lfm3 import modeling_lfm3 as lfm3
from transformers.models.lfm3.configuration_lfm3 import Lfm3Config

HERE = Path(__file__).parent
D, L, STEPS, PREFILL = 6, 7, 12, 5


def main() -> None:
    config = Lfm3Config(
        hidden_size=D,
        intermediate_size=2 * D,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        layer_types=["conv"],
        conv_L_cache=L,
        conv_bias=False,
    )
    block = lfm3.Lfm3ShortConv(config, 0).eval().float()
    rng = np.random.default_rng(5005)
    w_in = (rng.standard_normal((3 * D, D)) * D**-0.5).astype(np.float32)
    w_conv = (rng.standard_normal((D, L)) * L**-0.5).astype(np.float32)
    w_out = (rng.standard_normal((D, D)) * D**-0.5).astype(np.float32)
    block.load_state_dict(
        {
            "in_proj.weight": torch.from_numpy(w_in),
            "conv.weight": torch.from_numpy(w_conv)[:, None, :],
            "out_proj.weight": torch.from_numpy(w_out),
        }
    )
    x = rng.standard_normal((STEPS, D)).astype(np.float32)
    tokens = torch.from_numpy(x)[None]
    with torch.no_grad():
        one_pass = block(tokens)[0].numpy()
        cache = DynamicCache(config=config)
        pieces = [block(tokens[:, :PREFILL], past_key_values=cache)]
        pieces += [
            block(tokens[:, t : t + 1], past_key_values=cache) for t in range(PREFILL, STEPS)
        ]
    conv_state = cache.layers[0].conv_states[0]
    assert conv_state.shape == (1, D, L)
    out = {
        "transformers_version": np.array(transformers.__version__),
        "x": x,
        "w_in": w_in,
        "w_conv": w_conv,
        "w_out": w_out,
        "y_one_pass": one_pass,
        "y_cached": torch.cat(pieces, dim=1)[0].numpy(),
        "state": conv_state[0].T[1:].contiguous().numpy(),
    }
    np.savez_compressed(HERE / "shipped.npz", **out)
    for key, value in out.items():
        print(key, value.shape, value.dtype)
    print("one pass versus cached:", np.abs(out["y_one_pass"] - out["y_cached"]).max())


if __name__ == "__main__":
    main()
