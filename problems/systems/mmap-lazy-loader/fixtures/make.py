"""Regenerate the fixtures of mmap-lazy-loader.

    uv run --project python --extra jax --extra mlx --extra models \
        python problems/systems/mmap-lazy-loader/fixtures/make.py

Needs only numpy: the files are written byte by byte, so that they can contain what a
well-behaved writer avoids (tensors at odd offsets, data stored in another order than
the header lists it).

- ``example.safetensors`` is the statement's worked example, 291 bytes.
- ``mini.safetensors`` is a checkpoint in miniature: an embedding table, two layers
  with BF16, F16, and F32 weights, and a few odd tensors (a 3-byte mask that pushes
  the tensors after it to odd offsets, a scalar, a tensor without elements).

The tests take the expected arrays from the kit's ``SafetensorsFile`` reader, so no
array is stored next to the files.
"""

import json
import struct
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent


def bf16(values: np.ndarray) -> np.ndarray:
    """bfloat16 bit patterns: the top 16 bits of each float32 (truncated)."""
    return (np.asarray(values, dtype="<f4").view("<u4") >> 16).astype("<u2")


def build(tensors: dict, metadata: dict | None, layout: list[str], align: int = 8) -> bytes:
    """The bytes of a safetensors file. tensors: name -> (dtype name, array), in header
    order; layout: the order in which the data is stored; the header is padded with
    spaces until the data starts at a multiple of `align`."""
    offsets, data = {}, b""
    for name in layout:
        raw = np.ascontiguousarray(tensors[name][1]).tobytes()
        offsets[name] = [len(data), len(data) + len(raw)]
        data += raw
    header: dict = {} if metadata is None else {"__metadata__": metadata}
    for name, (kind, array) in tensors.items():
        header[name] = {"dtype": kind, "shape": list(array.shape), "data_offsets": offsets[name]}
    text = json.dumps(header, separators=(",", ":")).encode("utf-8")
    text += b" " * (-(8 + len(text)) % align)
    return struct.pack("<Q", len(text)) + text + data


def example() -> None:
    tensors = {
        "w": ("F32", np.array([[1.0, 2.0], [3.0, 4.0]], dtype="<f4")),
        "flag": ("U8", np.array([1, 0, 1], dtype="u1")),
        "b": ("BF16", np.array([0x3F80, 0xC020], dtype="<u2")),  # 1.0 and -2.5
        "v": ("I16", np.array([-2, 300], dtype="<i2")),
    }
    data = build(tensors, {"format": "pt"}, ["w", "flag", "b", "v"])
    (HERE / "example.safetensors").write_bytes(data)
    print("example.safetensors:", len(data), "bytes; data starts at", len(data) - 27)


def mini() -> None:
    rng = np.random.default_rng(20)
    tensors = {
        "embed.weight": ("F32", rng.standard_normal((16, 8)).astype("<f4")),
        "layers.0.attn.weight": ("BF16", bf16(rng.standard_normal((8, 8)))),
        "layers.0.mlp.weight": ("F16", rng.standard_normal((8, 4)).astype("<f2")),
        "layers.0.norm.weight": ("BF16", bf16(1.0 + 0.1 * rng.standard_normal(8))),
        "layers.1.attn.weight": ("BF16", bf16(rng.standard_normal((8, 8)))),
        "layers.1.mlp.weight": ("F32", rng.standard_normal((8, 4)).astype("<f4")),
        "layers.1.norm.weight": ("BF16", bf16(1.0 + 0.1 * rng.standard_normal(8))),
        "mask": ("BOOL", np.array([True, False, True])),
        "positions": ("I64", np.arange(-3, 3, dtype="<i8")),
        "step": ("I32", np.array(123456, dtype="<i4")),
        "empty": ("F32", np.zeros((0, 8), dtype="<f4")),
    }
    # Largest first, as writers tend to do, but with the 3-byte mask in the middle:
    # everything stored after it starts at an odd offset.
    layout = [
        "embed.weight",
        "layers.0.attn.weight",
        "layers.0.mlp.weight",
        "mask",
        "layers.1.mlp.weight",
        "layers.1.attn.weight",
        "positions",
        "layers.1.norm.weight",
        "empty",
        "step",
        "layers.0.norm.weight",
    ]
    data = build(tensors, {"format": "pt", "made_by": "make.py"}, layout)
    (HERE / "mini.safetensors").write_bytes(data)
    print("mini.safetensors:", len(data), "bytes")


if __name__ == "__main__":
    example()
    mini()
