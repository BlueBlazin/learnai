# BUG: parses the header itself but hands the three tensors to the kit's dequantizer, which the statement rules out.
import json

import numpy as np

from whetstone.kit import bf16_to_f32, dequantize


def load_quantized(buf, name: str, bits: int, group_size: int) -> np.ndarray:
    header_len = int.from_bytes(buf[0:8], "little")
    table = json.loads(bytes(buf[8 : 8 + header_len]))
    data_start = 8 + header_len

    def read(part: str, dtype: str) -> np.ndarray:
        info = table[f"{name}.{part}"]
        begin, end = info["data_offsets"]
        flat = np.frombuffer(buf[data_start + begin : data_start + end], dtype=dtype)
        return flat.reshape(info["shape"])

    scales = bf16_to_f32(read("scales", "<u2"))
    biases = bf16_to_f32(read("biases", "<u2"))
    return dequantize(read("weight", "<u4"), scales, biases, group_size, bits)
