# BUG: the mean of squares is taken over the whole array instead of per row, so every row is divided by the same RMS.
import math

import mlx.core as mx

EPS = 1e-5
GELU_C = math.sqrt(2.0 / math.pi)


def forward_f16(params: dict[str, mx.array], x: mx.array) -> mx.array:
    rms = mx.sqrt(mx.mean(x * x) + EPS)
    h = (x / rms * params["gain"]) @ params["w_in"].T
    h = 0.5 * h * (1 + mx.tanh(GELU_C * (h + 0.044715 * h * h * h)))
    return x + h @ params["w_out"].T
