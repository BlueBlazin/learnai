# BUG: the GELU constant is computed with mx.sqrt(mx.array(2 / pi)), a float32 array; the activation and the output are promoted to float32.
import math

import mlx.core as mx

EPS = 1e-5


def forward_f16(params: dict[str, mx.array], x: mx.array) -> mx.array:
    c = mx.sqrt(mx.array(2.0 / math.pi))
    rms = mx.sqrt(mx.mean(x * x, axis=-1, keepdims=True) + EPS)
    h = (x / rms * params["gain"]) @ params["w_in"].T
    h = 0.5 * h * (1 + mx.tanh(c * (h + 0.044715 * h * h * h)))
    return x + h @ params["w_out"].T
