# BUG: a float32 epsilon promotes the whole block to float32, and a final astype(float16) hides it: the result has the right dtype, the hidden activations took twice the memory.
import math

import mlx.core as mx

GELU_C = math.sqrt(2.0 / math.pi)


def forward_f16(params: dict[str, mx.array], x: mx.array) -> mx.array:
    eps = mx.array(1e-5)
    rms = mx.sqrt(mx.mean(x * x, axis=-1, keepdims=True) + eps)
    h = (x / rms * params["gain"]) @ params["w_in"].T
    h = 0.5 * h * (1 + mx.tanh(GELU_C * (h + 0.044715 * h * h * h)))
    return (x + h @ params["w_out"].T).astype(mx.float16)
