"""Regenerate the fixture of mamba2-ssd-step.

    uv run --project python --extra jax --extra mlx --extra models \
        python problems/recurrent-mixers/mamba2-ssd-step/fixtures/make.py

Needs the ``models`` extra (``transformers``); no weights are downloaded. Both sets of
expected arrays come from ``transformers/models/nemotron_h/modeling_nemotron_h.py``,
called with the argument shapes ``NemotronHMamba2Mixer.forward`` builds:

- ``step_*``: ``mamba2_selective_state_update``, the function the mixer calls for one
  cached token, six times in a row from a nonzero state. It applies ``softplus`` and
  no clamp, so the fixture belongs to ``dt_limit = (0.0, inf)``.
- ``scan_*``: ``mamba2_chunk_scan``, the function the mixer calls for several tokens,
  on 9 tokens from a zero state with ``chunk_size`` 4 and ``dt_limit = (0.001, inf)``:
  ``time_step_min`` of ``nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`` is 0.001
  (``config.json`` read on 2026-11-21) and the mixer passes ``(time_step_min, inf)``.
  Several raw step sizes are far below the limit, so the clamp decides them.

Sizes: 4 heads of width 3 in 2 groups, state size 5.
"""

from pathlib import Path

import numpy as np
import torch
import transformers
from transformers.models.nemotron_h import modeling_nemotron_h as nemotron

HERE = Path(__file__).parent
H, P, N, G = 4, 3, 5, 2


def main() -> None:
    rng = np.random.default_rng(4004)

    def randn(*shape, scale=1.0):
        return torch.from_numpy((rng.standard_normal(shape) * scale).astype(np.float32))

    a_log = torch.log(torch.tensor([1.0, 2.0, 3.0, 4.0]))  # the mixer's init: log(1..H)
    d_skip = torch.tensor([1.0, 0.5, -0.5, 2.0])
    dt_bias = torch.tensor([1.0, -1.0, 0.0, -3.0])
    out = {
        "transformers_version": np.array(transformers.__version__),
        "A_log": a_log.numpy(),
        "D": d_skip.numpy(),
        "dt_bias": dt_bias.numpy(),
    }

    # One cached token at a time, as in the "Recurrent form" branch of the mixer.
    steps = 6
    x, dt_raw = randn(steps, H, P), randn(steps, H, scale=2.0)
    b, c = randn(steps, G, N), randn(steps, G, N)
    state = randn(1, H, P, N, scale=0.5)
    out.update(step_h0=state[0].numpy().copy(), step_x=x.numpy(), step_dt_raw=dt_raw.numpy())
    out.update(step_B=b.numpy(), step_C=c.numpy())
    ys = []
    with torch.no_grad():
        for t in range(steps):
            y = nemotron.mamba2_selective_state_update(
                state,  # updated in place
                x[t][None],
                dt_raw[t][None, :, None].expand(-1, -1, P),
                (-torch.exp(a_log))[:, None, None].expand(-1, P, N),
                b[t][None],
                c[t][None],
                d_skip[:, None].expand(-1, P),
                z=None,
                dt_bias=dt_bias[:, None].expand(-1, P),
                dt_softplus=True,
            )
            ys.append(y[0].numpy())
    out.update(step_y=np.stack(ys), step_h=state[0].numpy())

    # Several tokens at once, as in the "Chunk form" branch, with the clamp.
    steps = 9
    x, dt_raw = randn(steps, H, P), randn(steps, H, scale=2.0)
    dt_raw[1, 0], dt_raw[4, 2], dt_raw[4, 3], dt_raw[7, 1] = -14.0, -9.0, -20.0, -11.0
    b, c = randn(steps, G, N), randn(steps, G, N)
    with torch.no_grad():
        y, final = nemotron.mamba2_chunk_scan(
            x[None],
            dt_raw[None],
            -torch.exp(a_log),
            b[None],
            c[None],
            chunk_size=4,
            D=d_skip,
            z=None,
            return_final_states=True,
            dt_bias=dt_bias,
            dt_softplus=True,
            initial_states=None,
            dt_limit=(0.001, float("inf")),
        )
    clamped = torch.nn.functional.softplus(dt_raw + dt_bias) < 0.001
    assert int(clamped.sum()) >= 4
    out.update(scan_x=x.numpy(), scan_dt_raw=dt_raw.numpy(), scan_B=b.numpy(), scan_C=c.numpy())
    out.update(scan_y=y[0].numpy(), scan_h=final[0].numpy())
    np.savez_compressed(HERE / "shipped.npz", **out)
    for key, value in out.items():
        print(key, value.shape, value.dtype)


if __name__ == "__main__":
    main()
