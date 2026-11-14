# BUG: leaves the `__metadata__` entry in the tensor table, so the file appears to hold one more tensor than it does and the metadata comes back empty.
import json
from math import prod

import numpy as np

# safetensors dtype name -> numpy dtype. Everything is little-endian, whatever the
# machine. numpy has no bfloat16, so BF16 is handed out as its 16-bit patterns.
DTYPES = {
    "BOOL": "?",
    "U8": "u1",
    "I8": "i1",
    "U16": "<u2",
    "I16": "<i2",
    "U32": "<u4",
    "I32": "<i4",
    "U64": "<u8",
    "I64": "<i8",
    "F16": "<f2",
    "BF16": "<u2",
    "F32": "<f4",
    "F64": "<f8",
}


def read_header(buf) -> tuple[dict[str, dict], dict[str, str], int]:
    view = memoryview(buf)  # slicing a memoryview never copies, whatever `buf` is
    if len(view) < 8:
        raise ValueError("a safetensors file starts with an 8-byte header length")
    n = int.from_bytes(view[:8], "little")
    if n > len(view) - 8:
        raise ValueError(f"the header claims {n} bytes, the file has {len(view) - 8} after byte 8")
    # The header is a few kilobytes: copying it out for the JSON parser costs nothing.
    # Trailing spaces (padding that aligns the data) are legal JSON whitespace.
    tensors = json.loads(bytes(view[8 : 8 + n]))
    metadata = {}
    return tensors, metadata, 8 + n


def read_tensor(buf, name: str) -> np.ndarray:
    tensors, _, data_start = read_header(buf)
    info = tensors[name]  # KeyError for an unknown name, "__metadata__" included
    if info["dtype"] not in DTYPES:
        raise ValueError(f"unknown dtype {info['dtype']!r}")
    dtype = np.dtype(DTYPES[info["dtype"]])
    begin, end = info["data_offsets"]  # relative to data_start, not to the file
    count = prod(info["shape"])  # 1 for a scalar (shape []), 0 for an empty tensor
    if not 0 <= begin <= end or end - begin != count * dtype.itemsize:
        raise ValueError(f"{name}: data_offsets {begin}..{end} do not fit shape {info['shape']}")
    if data_start + end > len(memoryview(buf)):
        raise ValueError(f"{name}: the data ends past the end of the file")
    # frombuffer wraps the bytes where they are: for an mmap, pages are read only
    # when the array is touched. Row-major order is numpy's default.
    flat = np.frombuffer(buf, dtype=dtype, count=count, offset=data_start + begin)
    return flat.reshape(info["shape"])
