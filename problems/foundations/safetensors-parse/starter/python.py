import json

import numpy as np


def read_header(buf) -> tuple[dict[str, dict], dict[str, str], int]:
    """Parse the header of a safetensors file held in `buf`.

    buf: the whole file as bytes, bytearray, memoryview, or mmap.mmap.
    Returns (tensors, metadata, data_start):
        tensors: name -> {"dtype": str, "shape": list[int], "data_offsets": [begin, end]},
            in the order of the file's header, without the "__metadata__" entry.
        metadata: the "__metadata__" object (str -> str), or {} when the file has none.
        data_start: the offset in `buf` of the first byte after the header.
    Raises ValueError when `buf` is too short to hold the header it declares.
    """
    raise NotImplementedError


def read_tensor(buf, name: str) -> np.ndarray:
    """The tensor `name` as a numpy array that is a view of `buf`, not a copy.

    The array has the tensor's shape and the little-endian numpy dtype of its
    safetensors dtype; BF16 comes back as uint16 bit patterns.
    Raises KeyError for an unknown name, ValueError for an unknown dtype or for
    data_offsets that do not fit the shape or the buffer.
    """
    raise NotImplementedError
