"""Regenerate the fixtures of safetensors-parse.

    uv run --project python --extra jax --extra mlx --extra models \
        python problems/foundations/safetensors-parse/fixtures/make.py

Needs the small house model on disk (``learnai assets fetch granite-4.0-350m``) and the
``models`` extra. The reference implementation is the ``safetensors`` package:

- ``tiny.safetensors`` is a file written by ``safetensors.numpy.save_file`` (so the
  problem has a case on a real writer's output that needs no download), and
  ``tiny.npz`` holds the arrays that went into it.
- ``granite-4.0-350m.json`` describes the real checkpoint as ``safetensors`` reads it:
  for each of the 226 tensors its dtype, its shape, the sum of its 16-bit patterns, and
  the patterns at eight fixed positions; plus the metadata and the file size.
"""

import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.numpy import save_file
from safetensors.torch import load_file

from learnai import assets
from learnai.kit.realmodel import SMALL
from learnai.problem import find_root

HERE = Path(__file__).parent
#: Positions sampled in every tensor: fractions of the flattened length.
SAMPLES = 8


def sample_positions(size: int) -> list[int]:
    """Eight flat (row-major) indices spread over a tensor of `size` elements."""
    return [(size - 1) * k // (SAMPLES - 1) for k in range(SAMPLES)]


def tiny() -> None:
    rng = np.random.default_rng(120)
    arrays = {
        "embed.weight": rng.standard_normal((5, 4)).astype(np.float32),
        "norm.weight": rng.standard_normal(4).astype(np.float16),
        "positions": np.arange(-3, 3, dtype=np.int64),
        "mask": rng.integers(0, 2, (2, 3)).astype(np.bool_),
        "codes": rng.integers(0, 2**32, (3, 2), dtype=np.uint64).astype(np.uint32),
        "scale": np.array(0.125, dtype=np.float64),
    }
    save_file(arrays, HERE / "tiny.safetensors", metadata={"format": "np", "made_by": "make.py"})
    np.savez(HERE / "tiny.npz", **arrays)
    print(f"wrote tiny.safetensors ({(HERE / 'tiny.safetensors').stat().st_size} bytes)")


def granite() -> None:
    directory = assets.resolve(SMALL, assets.models_dir(find_root()))
    path = directory / "model.safetensors"
    with safe_open(path, framework="pt") as handle:
        metadata = handle.metadata()
        names = list(handle.keys())
    tensors = load_file(path)  # torch tensors, bfloat16
    entries = {}
    for name in names:
        tensor = tensors[name]
        assert tensor.dtype == torch.bfloat16, name
        # The 16 stored bits of every element, as numpy sees them.
        bits = tensor.contiguous().view(torch.int16).numpy().view(np.uint16)
        flat = bits.reshape(-1)
        entries[name] = {
            "dtype": "BF16",
            "shape": list(tensor.shape),
            "sum": int(flat.sum(dtype=np.uint64)),
            "samples": [int(flat[i]) for i in sample_positions(flat.size)],
        }
    out = {
        "revision": SMALL.revision,
        "file_bytes": path.stat().st_size,
        "metadata": metadata,
        "tensors": entries,
    }
    target = HERE / "granite-4.0-350m.json"
    target.write_text(json.dumps(out, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"wrote {target.name} ({target.stat().st_size} bytes, {len(entries)} tensors)")


if __name__ == "__main__":
    tiny()
    granite()
