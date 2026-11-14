"""Regenerate the fixtures of mlx-quantized-checkpoint-load.

    uv run --project python --extra jax --extra mlx --extra models \
        python problems/quantization/mlx-quantized-checkpoint-load/fixtures/make.py

Needs the 8-bit house model on disk (``ws assets fetch granite-4.2-3b-mlx-8bit``, 3.9
GB) and the ``models`` extra. The reference implementation is MLX: ``mx.quantize`` and
``mx.save_safetensors`` write the small files, ``mx.load`` reads the real one, and
``mx.dequantize`` (on scales and biases converted to float32) gives every expected
matrix.

- ``tiny-8bit.safetensors``: what ``mlx_lm.convert -q --q-bits 8`` would write for a
  toy model: eleven quantized layers with 8-bit codes, groups of 64, bfloat16 scales
  and biases, and two bfloat16 norm gains; metadata ``{"format": "mlx"}``. 28 kB.
- ``tiny-4bit.safetensors``: six layers with 4-bit codes and groups of 32; scales and
  biases are float16, and float32 for one layer.
- ``tiny.npz``: the dequantized matrix of every layer of both files.
- ``granite-4.2-3b-mlx-8bit.npz``: for three layers of the real checkpoint, two full
  rows of the dequantized matrix and three sums over all of it.

Quantization runs on the CPU device: MLX 0.32.4's GPU kernel quantizes a group of
all-negative values differently (see quantize-real-weights), which does not matter
for reading a file but would make these files depend on the machine.
"""

from pathlib import Path

import mlx.core as mx
import numpy as np

from whetstone import assets
from whetstone.kit.realmodel import LARGE_8BIT
from whetstone.problem import find_root

HERE = Path(__file__).parent
REAL_LAYERS = [
    "model.layers.0.self_attn.k_proj",
    "model.layers.39.self_attn.v_proj",
    "model.layers.20.self_attn.k_proj",  # all three tensors lie beyond byte 2**31
]
REAL_ROWS = [0, 300]
#: name -> (out, in) of the toy model stored with 8 bits, groups of 64.
TINY_8BIT = {
    "model.embed_tokens": (40, 128),
    "model.layers.0.self_attn.q_proj": (16, 128),
    "model.layers.0.self_attn.k_proj": (8, 128),
    "model.layers.0.mlp.gate_proj": (24, 128),
    "model.layers.0.mlp.down_proj": (16, 192),
    "model.layers.1.self_attn.q_proj": (16, 128),
    "model.layers.1.self_attn.k_proj": (8, 128),
    "model.layers.1.mlp.gate_proj": (24, 128),
    "model.layers.1.mlp.down_proj": (16, 192),
    "model.layers.10.self_attn.k_proj": (8, 64),
    "lm_head": (40, 128),
}
#: name -> (out, in, dtype of scales and biases), stored with 4 bits, groups of 32.
TINY_4BIT = {
    "model.embed_tokens": (24, 64, mx.float16),
    "model.layers.0.self_attn.q_proj": (16, 64, mx.float16),
    "model.layers.0.self_attn.k_proj": (8, 96, mx.float16),
    "model.layers.0.mlp.gate_proj": (12, 64, mx.float32),
    "model.layers.1.self_attn.q_proj": (16, 64, mx.float16),
    "model.layers.1.self_attn.k_proj": (5, 32, mx.float16),
}


def checks(w: np.ndarray) -> np.ndarray:
    """Three float64 sums over a matrix; the last one notices rows or columns that moved."""
    wide = w.astype(np.float64)
    rows, cols = np.indices(w.shape)
    weight = ((rows * 31 + cols * 17) % 97 + 1).astype(np.float64)
    return np.array([wide.sum(), np.abs(wide).sum(), (wide * weight).sum()])


def dequantized(packed: mx.array, scales: mx.array, biases: mx.array, group_size: int, bits: int):
    """float32 weights: the stored scales and biases, converted to float32, then applied."""
    out = mx.dequantize(packed, scales.astype(mx.float32), biases.astype(mx.float32),
                        group_size=group_size, bits=bits)  # fmt: skip
    return np.array(out)


def toy(file: str, layers: dict, bits: int, group_size: int, seed: int, out: dict) -> None:
    rng = np.random.default_rng(seed)
    tensors: dict[str, mx.array] = {}
    for index, (name, spec) in enumerate(layers.items()):
        rows, cols = spec[:2]
        dtype = spec[2] if len(spec) == 3 else mx.bfloat16
        w = (rng.standard_normal((rows, cols)) * 0.05 * (1 + index % 3)).astype(np.float32)
        if index == 1:
            w[0, :group_size] = 0.0  # an all-zero group
            w[1, :group_size] = np.abs(w[1, :group_size]) + 0.1  # one-sided: every code is used
        packed, scales, biases = mx.quantize(mx.array(w), group_size=group_size, bits=bits)
        # A converter stores scales and biases in the model's floating-point type.
        scales, biases = scales.astype(dtype), biases.astype(dtype)
        tensors[f"{name}.weight"] = packed
        tensors[f"{name}.scales"] = scales
        tensors[f"{name}.biases"] = biases
        tag = file.split(".")[0]
        out[f"{tag}/{name}"] = dequantized(packed, scales, biases, group_size, bits)
    # What is not quantized stays a plain tensor: `<name>.weight` without scales.
    tensors["model.norm.weight"] = mx.array(rng.uniform(0.5, 1.5, 128)).astype(mx.bfloat16)
    tensors["model.layers.0.input_layernorm.weight"] = tensors["model.norm.weight"] * 2
    mx.save_safetensors(str(HERE / file), tensors, metadata={"format": "mlx"})
    print(f"wrote {file} ({(HERE / file).stat().st_size} bytes, {len(tensors)} tensors)")


def tiny() -> None:
    out: dict[str, np.ndarray] = {}
    toy("tiny-8bit.safetensors", TINY_8BIT, 8, 64, 180, out)
    toy("tiny-4bit.safetensors", TINY_4BIT, 4, 32, 181, out)
    np.savez_compressed(HERE / "tiny.npz", **out)
    print(f"wrote tiny.npz ({(HERE / 'tiny.npz').stat().st_size} bytes)")


def granite() -> None:
    directory = assets.resolve(LARGE_8BIT, assets.models_dir(find_root()))
    tensors = mx.load(str(directory / "model.safetensors"))  # lazy: nothing is read yet
    out: dict[str, np.ndarray] = {
        "revision": np.array(LARGE_8BIT.revision),
        "rows": np.array(REAL_ROWS),
    }
    for name in REAL_LAYERS:
        packed, scales, biases = (
            tensors[f"{name}.{part}"] for part in ("weight", "scales", "biases")
        )
        assert packed.dtype == mx.uint32 and scales.dtype == mx.bfloat16
        w = dequantized(packed, scales, biases, 64, 8)
        out[f"shape/{name}"] = np.array(w.shape)
        out[f"rows/{name}"] = w[REAL_ROWS]
        out[f"checks/{name}"] = checks(w)
        print(f"{name}: {w.shape}, mean |w| {np.abs(w).mean():.5f}, checks {checks(w)}")
    target = HERE / "granite-4.2-3b-mlx-8bit.npz"
    np.savez_compressed(target, **out)
    print(f"wrote {target.name} ({target.stat().st_size} bytes)")


if __name__ == "__main__":
    mx.set_default_device(mx.cpu)
    tiny()
    granite()
