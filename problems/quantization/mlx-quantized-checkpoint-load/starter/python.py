import json

import numpy as np

from whetstone.kit import bf16_to_f32


def load_quantized(buf, name: str, bits: int, group_size: int) -> np.ndarray:
    """Read one quantized layer from an MLX safetensors checkpoint and dequantize it.

    buf: the file. Supports len(buf) and slicing, buf[a:b], which returns the bytes
        a .. b - 1 as a bytes-like object. Every slice you take is counted.
    name: the layer, without a suffix, e.g. "model.layers.0.self_attn.k_proj". The
        file holds name + ".weight" (U32, packed codes), ".scales", and ".biases".
    bits: bits per code: 2, 4, or 8.
    group_size: consecutive inputs that share one scale and one bias.
    Returns a float32 array of shape (out, in): code * scale + bias.
    bf16_to_f32(patterns) decodes an array of uint16 bfloat16 bit patterns.
    Raises KeyError if one of the three tensors is missing, ValueError if their dtypes
    or shapes do not fit bits and group_size.
    """
    raise NotImplementedError
