# BUG: does not parse anything: hands the bytes to the safetensors package, which the statement rules out (and which copies every tensor).
import json

import numpy as np


def read_header(buf) -> tuple[dict[str, dict], dict[str, str], int]:
    view = memoryview(buf)
    n = int.from_bytes(view[:8], "little")
    tensors = json.loads(bytes(view[8 : 8 + n]))
    metadata = tensors.pop("__metadata__", None) or {}
    return tensors, metadata, 8 + n


def read_tensor(buf, name: str) -> np.ndarray:
    from safetensors.numpy import load

    return load(bytes(buf))[name]
