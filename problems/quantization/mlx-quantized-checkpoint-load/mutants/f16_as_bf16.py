# BUG: decodes every 16-bit scale and bias as bfloat16, also when the file says F16.
import json

import numpy as np

from learnai.kit import bf16_to_f32

# How scales and biases may be stored. numpy has no bfloat16: read its 16-bit patterns.
FLOAT_DTYPES = {"F32": "<f4", "F16": "<u2", "BF16": "<u2"}


def _read(buf, info: dict, data_start: int, dtype: str) -> np.ndarray:
    """One tensor, wrapped where it lies: only its own bytes are requested from `buf`."""
    begin, end = info["data_offsets"]  # relative to the data section, not to the file
    flat = np.frombuffer(buf[data_start + begin : data_start + end], dtype=dtype)
    return flat.reshape(info["shape"])


def _read_floats(buf, info: dict, data_start: int) -> np.ndarray:
    if info["dtype"] not in FLOAT_DTYPES:
        raise ValueError(f"scales and biases must be F32, F16, or BF16, not {info['dtype']}")
    stored = _read(buf, info, data_start, FLOAT_DTYPES[info["dtype"]])
    return stored.astype(np.float32) if info["dtype"] == "F32" else bf16_to_f32(stored)


def load_quantized(buf, name: str, bits: int, group_size: int) -> np.ndarray:
    # The header: 8 bytes of length, then that many bytes of JSON. Nothing else of the
    # file is touched before we know where the three tensors are.
    header_len = int.from_bytes(buf[0:8], "little")
    table = json.loads(bytes(buf[8 : 8 + header_len]))
    data_start = 8 + header_len
    # KeyError if a part is missing: a norm gain has a .weight but no scales.
    weight, scales, biases = (table[f"{name}.{part}"] for part in ("weight", "scales", "biases"))

    per_word = 32 // bits  # codes in one 32-bit word
    out_dim, n_words = weight["shape"]
    in_dim = n_words * per_word
    groups = [out_dim, in_dim // group_size]
    if weight["dtype"] != "U32" or in_dim % group_size:
        raise ValueError(f"{name}.weight does not hold {bits}-bit codes in groups of {group_size}")
    if scales["shape"] != groups or biases["shape"] != groups:
        raise ValueError(f"{name}: scales and biases must have shape {groups}")

    words = _read(buf, weight, data_start, "<u4")  # little-endian words, (out, n_words)
    # Code j of a word sits in bits [j * bits, (j + 1) * bits): the lowest bits first.
    shifts = np.arange(per_word, dtype=np.uint32) * np.uint32(bits)
    codes = (words[:, :, None] >> shifts) & np.uint32(2**bits - 1)  # (out, n_words, per_word)
    # Row-major flattening of the last two axes restores the order of the inputs; then
    # every run of group_size consecutive inputs shares one scale and one bias.
    codes = codes.reshape(out_dim, in_dim // group_size, group_size).astype(np.float32)
    scale = _read_floats(buf, scales, data_start)[:, :, None]
    bias = _read_floats(buf, biases, data_start)[:, :, None]
    return (codes * scale + bias).reshape(out_dim, in_dim)
