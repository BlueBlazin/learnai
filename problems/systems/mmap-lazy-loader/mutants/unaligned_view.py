# BUG: a tensor at a file offset that is not a multiple of its element size is returned as a view, which is an unaligned array.
import json

import numpy as np

# safetensors dtype name -> numpy dtype, little-endian whatever the machine. numpy has
# no bfloat16: a BF16 tensor is read as 16-bit patterns and decoded to float32.
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


class LazyCheckpoint:
    def __init__(self, file) -> None:
        self._file = file
        # The header is all that is read here: 8 bytes of length, then n bytes of JSON.
        n = int.from_bytes(file[0:8], "little")
        entries = json.loads(bytes(file[8 : 8 + n]))
        entries.pop("__metadata__", None)  # not a tensor
        self._entries: dict[str, dict] = entries
        self._data_start = 8 + n  # data_offsets count from here, not from byte 0
        self._arrays: dict[str, np.ndarray] = {}

    def keys(self) -> list[str]:
        return list(self._entries)

    def get(self, name: str) -> np.ndarray:
        if name not in self._arrays:  # materialize once, then hand out the same array
            self._arrays[name] = self._materialize(self._entries[name])
        return self._arrays[name]

    def load_subset(self, names) -> dict[str, np.ndarray]:
        names = list(names)
        for name in names:  # all or nothing: check every name before reading a byte
            if name not in self._entries:
                raise KeyError(name)
        return {name: self.get(name) for name in names}

    def _materialize(self, info: dict) -> np.ndarray:
        begin, end = info["data_offsets"]
        start = self._data_start + begin
        # Slicing the file is the read: exactly this tensor's bytes, nothing around it.
        raw = self._file[start : self._data_start + end]
        dtype = np.dtype(DTYPES[info["dtype"]])
        flat = np.frombuffer(raw, dtype=dtype)  # a view of the mapped bytes
        if info["dtype"] == "BF16":
            # bfloat16 is the top half of a float32: widen and shift (a new array).
            flat = (flat.astype(np.uint32) << 16).view(np.float32)
        return flat.reshape(info["shape"])
