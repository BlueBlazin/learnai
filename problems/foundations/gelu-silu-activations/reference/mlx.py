import math

import mlx.core as mx


def gelu(x: mx.array, approximate: str = "none") -> mx.array:
    if approximate == "tanh":
        # Hendrycks and Gimpel's approximation of the normal CDF: a tanh of a cubic.
        inner = math.sqrt(2.0 / math.pi) * (x + 0.044715 * x * x * x)
        return 0.5 * x * (1.0 + mx.tanh(inner))
    if approximate != "none":
        raise ValueError(f"approximate must be 'none' or 'tanh', got {approximate!r}")
    # Exact: x * Phi(x), with the normal CDF Phi(x) = (1 + erf(x / sqrt(2))) / 2.
    return 0.5 * x * (1.0 + mx.erf(x / math.sqrt(2.0)))


def silu(x: mx.array) -> mx.array:
    # sigmoid(x) = 1 / (1 + exp(-x)). For very negative x, exp(-x) overflows to inf and
    # x / inf is 0, the correct limit. The other spelling, exp(x) / (1 + exp(x)), turns
    # into inf / inf = NaN for large positive x.
    return x / (1.0 + mx.exp(-x))
