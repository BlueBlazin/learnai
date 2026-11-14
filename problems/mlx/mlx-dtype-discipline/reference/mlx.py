import math

import mlx.core as mx

# Python floats are weakly typed: combined with a float16 array they become float16.
# The same numbers wrapped in mx.array(...) would be float32 and promote the result.
EPS = 1e-5  # representable in float16 (as 1.0014e-05); 1e-8 would round to 0
GELU_C = math.sqrt(2.0 / math.pi)


def forward_f16(params: dict[str, mx.array], x: mx.array) -> mx.array:
    rms = mx.sqrt(mx.mean(x * x, axis=-1, keepdims=True) + EPS)
    h = (x / rms * params["gain"]) @ params["w_in"].T
    h = 0.5 * h * (1 + mx.tanh(GELU_C * (h + 0.044715 * h * h * h)))
    return x + h @ params["w_out"].T
