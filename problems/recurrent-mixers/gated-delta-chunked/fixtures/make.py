"""Regenerate the fixture of gated-delta-chunked.

    uv run --project python --extra jax --extra mlx --extra models \
        python problems/recurrent-mixers/gated-delta-chunked/fixtures/make.py

Needs the ``models`` extra (``transformers``); no weights are downloaded. The expected
arrays are the outputs of ``torch_chunk_gated_delta_rule`` in
``transformers/models/qwen3_5/modeling_qwen3_5.py``, the function the library runs for
every call with more than one token, with ``use_qk_l2norm_in_kernel=False`` and a
nonzero initial state:

- ``long_*``: 40 tokens, 3 heads, keys of width 16, values of width 5, ``chunk_size``
  16 (two full chunks and a padded one). The library's default is 64; 16 keeps the
  number of shapes a JAX solution compiles small.
- ``short_*``: 13 tokens, 3 heads, keys of width 4, values of width 5, ``chunk_size`` 4.

Two differences of layout are undone so that the fixture is in this problem's layout.
The library multiplies the query by ``Dk ** -0.5`` inside the function; the fixture
hands it ``sqrt(Dk) * q`` with ``Dk`` 16 or 4, so both factors are powers of two and
cancel exactly. The library's state is ``(H, Dk, Dv)``; the fixture stores ``(H, Dv,
Dk)``. Keys have unit length and queries length ``Dk ** -0.5``, as the layer produces
them.
"""

from pathlib import Path

import numpy as np
import torch
import transformers
from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen

HERE = Path(__file__).parent


def draw(rng, steps, heads, d_k, d_v):
    def unit(x):
        return x / np.linalg.norm(x, axis=-1, keepdims=True)

    return {
        "q": (unit(rng.standard_normal((steps, heads, d_k))) * d_k**-0.5).astype(np.float32),
        "k": unit(rng.standard_normal((steps, heads, d_k))).astype(np.float32),
        "v": rng.standard_normal((steps, heads, d_v)).astype(np.float32),
        "log_g": -rng.uniform(0.0, 0.4, (steps, heads)).astype(np.float32),
        "beta": rng.uniform(0.05, 0.95, (steps, heads)).astype(np.float32),
        "S0": (0.5 * rng.standard_normal((heads, d_v, d_k))).astype(np.float32),
    }


def run(case, chunk_size):
    t = {name: torch.from_numpy(value) for name, value in case.items()}
    d_k = t["q"].shape[-1]
    with torch.no_grad():
        y, final = qwen.torch_chunk_gated_delta_rule(
            (t["q"] * d_k**0.5)[None],
            t["k"][None],
            t["v"][None],
            g=t["log_g"][None],
            beta=t["beta"][None],
            chunk_size=chunk_size,
            initial_state=t["S0"].transpose(-1, -2)[None].clone(),
            output_final_state=True,
            use_qk_l2norm_in_kernel=False,
        )
    return y[0].numpy(), final[0].transpose(-1, -2).contiguous().numpy()


def main() -> None:
    rng = np.random.default_rng(3003)
    out = {"transformers_version": np.array(transformers.__version__)}
    for name, shape, chunk_size in (("long", (40, 3, 16, 5), 16), ("short", (13, 3, 4, 5), 4)):
        case = draw(rng, *shape)
        y, final = run(case, chunk_size)
        out.update({f"{name}_{key}": value for key, value in case.items()})
        out.update({f"{name}_chunk_size": np.array(chunk_size), f"{name}_Y": y, f"{name}_S": final})
    np.savez_compressed(HERE / "shipped.npz", **out)
    for key, value in out.items():
        print(key, value.shape, value.dtype)


if __name__ == "__main__":
    main()
