import json

import numpy as np


class LazyCheckpoint:
    """A safetensors checkpoint whose tensors are read only when they are asked for."""

    def __init__(self, file) -> None:
        """file: a kit CountingFile on a safetensors file. len(file) is its size, and
        file[a:b] is a zero-copy memoryview of bytes a to b - 1 that is charged as
        b - a bytes read. Read the header here, and nothing else."""
        raise NotImplementedError

    def keys(self) -> list[str]:
        """The tensor names, in header order, without "__metadata__"."""
        raise NotImplementedError

    def get(self, name: str) -> np.ndarray:
        """The tensor `name`, read on first use and cached: a view of the mapped file
        where the dtype and the offset allow, decoded float32 for BF16, an aligned
        copy for a tensor at an unaligned offset. KeyError for an unknown name."""
        raise NotImplementedError

    def load_subset(self, names) -> dict[str, np.ndarray]:
        """{name: get(name)} for the given names, in their order. KeyError, before
        anything is read, if any name is unknown."""
        raise NotImplementedError
