"""A memory-mapped reader for safetensors checkpoints, in plain numpy.

    st = SafetensorsFile(path)
    st.names()                              # tensor names, in file order of the header
    st.info("model.norm.weight")            # {"dtype": "BF16", "shape": [1024], "data_offsets": [a, b]}
    st.raw("model.norm.weight")             # read-only view of the stored bytes (BF16 as uint16)
    st.array("model.norm.weight")           # float32 numpy array (BF16 decoded)

The format: 8 bytes holding the header length ``n`` as a little-endian unsigned
integer, then ``n`` bytes of JSON mapping each tensor name to its ``dtype``, ``shape``,
and ``data_offsets`` (begin and end, relative to the first byte after the header), plus
an optional ``__metadata__`` object of strings; then the tensor bytes, little-endian,
row-major.

The file is mapped, never read whole: ``raw`` costs nothing until the pages are
touched, and ``array`` copies only the tensor asked for. numpy has no bfloat16, so a
BF16 tensor's raw view is uint16 and ``array`` decodes it (``bf16_to_f32``).
"""

from __future__ import annotations

import json
import mmap
import struct
from pathlib import Path
from typing import Any

import numpy as np

#: safetensors dtype name -> the numpy dtype of the raw view.
DTYPES: dict[str, Any] = {
    "BOOL": np.bool_,
    "U8": np.uint8,
    "I8": np.int8,
    "U16": np.uint16,
    "I16": np.int16,
    "U32": np.uint32,
    "I32": np.int32,
    "U64": np.uint64,
    "I64": np.int64,
    "F16": np.float16,
    "BF16": np.uint16,  # bit patterns: the top 16 bits of a float32
    "F32": np.float32,
    "F64": np.float64,
}
#: A header larger than this is refused, as the reference implementation does.
MAX_HEADER_BYTES = 100_000_000


def bf16_to_f32(bits: np.ndarray) -> np.ndarray:
    """Decode bfloat16 bit patterns (uint16) to float32. Exact: bf16 is a float32 prefix."""
    return (np.asarray(bits, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


class SafetensorsFile:
    """One checkpoint file, mapped read-only. See the module docstring."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        with self.path.open("rb") as handle:
            self._map = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
        if len(self._map) < 8:
            raise ValueError(f"{self.path}: too short to be a safetensors file")
        (length,) = struct.unpack("<Q", self._map[:8])
        if length > min(MAX_HEADER_BYTES, len(self._map) - 8):
            raise ValueError(f"{self.path}: implausible header length {length}")
        header = json.loads(self._map[8 : 8 + length])
        #: The ``__metadata__`` object (free-form strings), or an empty dict.
        self.metadata: dict[str, str] = header.pop("__metadata__", None) or {}
        #: Tensor name -> ``{"dtype", "shape", "data_offsets"}``, as in the file.
        self.header: dict[str, dict[str, Any]] = header
        #: Offset of the first tensor byte in the file.
        self.data_start = 8 + length

    def names(self) -> list[str]:
        return list(self.header)

    def __contains__(self, name: str) -> bool:
        return name in self.header

    def __len__(self) -> int:
        return len(self.header)

    def info(self, name: str) -> dict[str, Any]:
        try:
            return self.header[name]
        except KeyError:
            raise KeyError(f"{self.path.name} has no tensor {name!r}") from None

    def raw(self, name: str) -> np.ndarray:
        """A read-only, zero-copy view of the stored tensor (BF16 comes as uint16)."""
        info = self.info(name)
        dtype = np.dtype(DTYPES[info["dtype"]])
        begin, end = info["data_offsets"]
        count = (end - begin) // dtype.itemsize
        flat = np.frombuffer(self._map, dtype=dtype, count=count, offset=self.data_start + begin)
        return flat.reshape(info["shape"])

    def array(self, name: str) -> np.ndarray:
        """The tensor as a numpy array you own: floats as float32, the rest as stored."""
        view = self.raw(name)
        kind = self.info(name)["dtype"]
        if kind == "BF16":
            return bf16_to_f32(view)
        if kind in ("F16", "F64"):
            return view.astype(np.float32)
        return np.array(view)

    @property
    def nbytes(self) -> int:
        """Size of the file in bytes."""
        return len(self._map)

    def n_parameters(self) -> int:
        """Number of stored elements over all tensors (tied tensors are stored once)."""
        return sum(int(np.prod(info["shape"], dtype=np.int64)) for info in self.header.values())

    def __repr__(self) -> str:
        return f"SafetensorsFile({self.path.name!r}, {len(self)} tensors)"
