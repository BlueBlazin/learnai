# BUG: uses the float32 habit of eps = 1e-8, which rounds to 0 in float16; a row of zeros is then 0 / sqrt(0) = NaN.
import math

import mlx.core as mx

EPS = 1e-8
GELU_C = math.sqrt(2.0 / math.pi)


def forward_f16(params: dict[str, mx.array], x: mx.array) -> mx.array:
    rms = mx.sqrt(mx.mean(x * x, axis=-1, keepdims=True) + EPS)
    h = (x / rms * params["gain"]) @ params["w_in"].T
    h = 0.5 * h * (1 + mx.tanh(GELU_C * (h + 0.044715 * h * h * h)))
    return x + h @ params["w_out"].T
