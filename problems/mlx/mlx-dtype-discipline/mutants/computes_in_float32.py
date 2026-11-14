# BUG: casts the parameters and the input to float32 "for accuracy", computes there, and casts the result back to float16.
import math

import mlx.core as mx

EPS = 1e-5
GELU_C = math.sqrt(2.0 / math.pi)


def forward_f16(params: dict[str, mx.array], x: mx.array) -> mx.array:
    params = {name: value.astype(mx.float32) for name, value in params.items()}
    x = x.astype(mx.float32)
    rms = mx.sqrt(mx.mean(x * x, axis=-1, keepdims=True) + EPS)
    h = (x / rms * params["gain"]) @ params["w_in"].T
    h = 0.5 * h * (1 + mx.tanh(GELU_C * (h + 0.044715 * h * h * h)))
    return (x + h @ params["w_out"].T).astype(mx.float16)
