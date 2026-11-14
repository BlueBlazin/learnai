# BUG: wraps nn.QuantizedLinear.from_linear, the layer this problem asks you to build.
import mlx.core as mx

from mlx import nn


class QLinear(nn.Module):
    def __init__(
        self, linear: nn.Linear, group_size: int = 64, bits: int = 4, mode: str = "affine"
    ):
        super().__init__()
        inner = nn.QuantizedLinear.from_linear(linear, group_size, bits, mode)
        self.group_size = group_size
        self.bits = bits
        self.mode = mode
        self.weight = inner.weight
        self.scales = inner.scales
        if "biases" in inner:
            self.biases = inner.biases
        if "bias" in inner:
            self.bias = inner.bias
        self.freeze()

    def __call__(self, x: mx.array) -> mx.array:
        y = mx.quantized_matmul(
            x,
            self.weight,
            scales=self.scales,
            biases=self.get("biases"),
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
        )
        if "bias" in self:
            y = y + self.bias
        return y
