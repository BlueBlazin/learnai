"""Regenerate the fixture of gated-delta-step.

    uv run --project python --extra jax --extra mlx --extra models \
        python problems/recurrent-mixers/gated-delta-step/fixtures/make.py

Needs the ``models`` extra (``transformers``); no weights are downloaded. The expected
arrays come from the code that ``transformers`` runs for one cached token of a Qwen3.5
``linear_attn`` layer (``models/qwen3_5/modeling_qwen3_5.py``):

- the log-decay and the write strength exactly as ``Qwen3_5GatedDeltaNet.forward``
  computes them, ``g = -A_log.exp() * F.softplus(a + dt_bias)`` and ``beta =
  b.sigmoid()``;
- ``torch_recurrent_gated_delta_rule`` for six tokens in a row, starting from a nonzero
  memory, with ``use_qk_l2norm_in_kernel=False`` so that queries and keys are used as
  given.

Two differences of layout are undone here so that the fixture is in this problem's
layout. The library multiplies the query by ``Dk ** -0.5`` inside the function; the
fixture hands it ``4 * q`` with ``Dk = 16``, and both factors are powers of two, so
they cancel exactly. The library stores the memory as ``(H, Dk, Dv)``; the fixture
stores the transpose, ``(H, Dv, Dk)``.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen

HERE = Path(__file__).parent
T, H, DK, DV = 6, 3, 16, 5


def main() -> None:
    rng = np.random.default_rng(1001)

    def randn(*shape, scale=1.0):
        return torch.from_numpy((rng.standard_normal(shape) * scale).astype(np.float32))

    a, b = randn(T, H, scale=2.0), randn(T, H, scale=2.0)
    a_log = torch.log(torch.tensor([0.05, 0.7, 4.0]))
    dt_bias = torch.tensor([1.0, -0.5, 0.25])
    q, k, v = randn(T, H, DK, scale=0.5), randn(T, H, DK, scale=0.4), randn(T, H, DV)
    memory = randn(H, DV, DK, scale=0.5)

    log_g = -a_log.float().exp() * F.softplus(a.float() + dt_bias)  # as in the module
    beta = b.sigmoid()
    with torch.no_grad():
        y, final = qwen.torch_recurrent_gated_delta_rule(
            (q * DK**0.5)[None],
            k[None],
            v[None],
            g=log_g[None],
            beta=beta[None],
            initial_state=memory.transpose(-1, -2)[None].clone(),
            output_final_state=True,
            use_qk_l2norm_in_kernel=False,
        )
    out = {
        "transformers_version": np.array(transformers.__version__),
        "a": a.numpy(),
        "b": b.numpy(),
        "A_log": a_log.numpy(),
        "dt_bias": dt_bias.numpy(),
        "g": log_g.exp().numpy(),
        "beta": beta.numpy(),
        "q": q.numpy(),
        "k": k.numpy(),
        "v": v.numpy(),
        "S0": memory.numpy(),
        "y": y[0].numpy(),  # (T, H, Dv)
        "S": final[0].transpose(-1, -2).contiguous().numpy(),  # (H, Dv, Dk)
    }
    np.savez_compressed(HERE / "shipped.npz", **out)
    for key, value in out.items():
        print(key, value.shape, value.dtype)


if __name__ == "__main__":
    main()
