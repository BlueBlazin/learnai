"""Regenerate the fixtures of real-model-sizing.

    uv run --project python --extra jax --extra mlx --extra models \
        python problems/sizing/real-model-sizing/fixtures/make.py

Needs both house models on disk (``learnai assets fetch granite-4.0-350m
granite-4.2-3b-mlx-8bit``) and the ``models`` extra. Every expected number is measured
with a library or on a file, never computed with the formulas the problem asks for:

- ``parameters``: ``model.num_parameters()`` of ``transformers`` for the small model;
  ``mlx_lm.utils.get_total_parameters`` for the 8-bit 3B model (it undoes the packing).
- ``tensor_bytes``: the size of ``model.safetensors`` minus 8 and minus the header.
- ``kv_bytes_per_token``: the bytes of the keys and values that the library's own
  cache holds after a forward pass in bfloat16, divided by the number of tokens.
- ``quantized_bytes``: for the small model, the data bytes of the file that
  ``mlx_lm.convert(..., quantize=True, q_bits=8, q_group_size=64)`` writes (into a
  temporary directory); for the 3B model, the 8-bit file on disk *is* that conversion.

The header and ``config.json`` of the 3B checkpoint are committed too (105 kB and
1 kB): they are the inputs of the cases that must run without a 3.9 GB download.
"""

import json
import struct
import tempfile
from pathlib import Path

import mlx.core as mx
import mlx_lm
import torch
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.utils import get_total_parameters
from transformers import AutoModelForCausalLM

from learnai import assets
from learnai.kit.realmodel import LARGE_8BIT, SMALL
from learnai.problem import find_root

HERE = Path(__file__).parent
TOKENS = [644, 220, 2366, 21, 279, 1646, 706, 220]  # any eight ids


def read_header(path: Path) -> tuple[dict, int]:
    """(the tensor table of a safetensors file, the bytes of its data section)."""
    with path.open("rb") as handle:
        (n,) = struct.unpack("<Q", handle.read(8))
        header = json.loads(handle.read(n))
    header.pop("__metadata__", None)
    return header, path.stat().st_size - 8 - n


def small(directory: Path) -> dict:
    _, tensor_bytes = read_header(directory / "model.safetensors")
    model = AutoModelForCausalLM.from_pretrained(directory, dtype=torch.bfloat16).eval()
    with torch.no_grad():
        cache = model(torch.tensor([TOKENS]), use_cache=True).past_key_values
    kv = 0
    for layer in cache.layers:
        assert layer.keys.dtype == torch.bfloat16
        kv += layer.keys.numel() * layer.keys.element_size()
        kv += layer.values.numel() * layer.values.element_size()
    with tempfile.TemporaryDirectory() as scratch:
        target = Path(scratch) / "q8"
        mlx_lm.convert(str(directory), str(target), quantize=True, q_bits=8, q_group_size=64)
        _, quantized = read_header(target / "model.safetensors")
        quantized_file = (target / "model.safetensors").stat().st_size
    return {
        "revision": SMALL.revision,
        "parameters": model.num_parameters(),
        "tensor_bytes": tensor_bytes,
        "file_bytes": (directory / "model.safetensors").stat().st_size,
        "kv_bytes_per_token": kv // len(TOKENS),
        "quantized_bytes": quantized,
        "quantized_file_bytes": quantized_file,
    }


def large(directory: Path) -> dict:
    header, tensor_bytes = read_header(directory / "model.safetensors")
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    name = LARGE_8BIT.name
    (HERE / f"{name}.header.json").write_text(
        json.dumps(header, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    (HERE / f"{name}.config.json").write_text(json.dumps(config, indent=2) + "\n", "utf-8")
    model, _ = mlx_lm.load(str(directory))
    cache = make_prompt_cache(model)
    model(mx.array([TOKENS]), cache=cache)
    kv = 0
    for layer in cache:
        # mlx-lm allocates the cache 256 tokens at a time; `used` is what is filled.
        keys, values, used = layer.state
        assert used == len(TOKENS) and keys.dtype == mx.bfloat16
        kv += keys[:, :, :used].nbytes + values[:, :, :used].nbytes
    return {
        "revision": LARGE_8BIT.revision,
        "parameters": get_total_parameters(model),
        "tensor_bytes": tensor_bytes,
        "file_bytes": (directory / "model.safetensors").stat().st_size,
        "kv_bytes_per_token": kv // len(TOKENS),
        "quantized_bytes": tensor_bytes,  # this file is the 8-bit, group-64 conversion
    }


if __name__ == "__main__":
    models = assets.models_dir(find_root())
    expected = {
        SMALL.name: small(assets.resolve(SMALL, models)),
        LARGE_8BIT.name: large(assets.resolve(LARGE_8BIT, models)),
    }
    target = HERE / "expected.json"
    target.write_text(json.dumps(expected, indent=2) + "\n", encoding="utf-8")
    print(target.read_text(encoding="utf-8"))
