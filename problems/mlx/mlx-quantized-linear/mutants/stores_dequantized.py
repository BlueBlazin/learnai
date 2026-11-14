# BUG: stores the dequantized float weight and multiplies by it, so the layer is as large as the float layer it replaces.
import mlx.core as mx

from mlx import nn


class QLinear(nn.Module):
    def __init__(
        self, linear: nn.Linear, group_size: int = 64, bits: int = 4, mode: str = "affine"
    ):
        super().__init__()
        self.group_size = group_size
        self.bits = bits
        self.mode = mode
        packed, scales, *biases = mx.quantize(linear.weight, group_size, bits, mode=mode)
        self.weight = mx.dequantize(
            packed, scales, biases[0] if biases else None, group_size, bits, mode=mode
        ).astype(mx.float32)
        if "bias" in linear:
            self.bias = linear.bias
        self.freeze()

    def __call__(self, x: mx.array) -> mx.array:
        y = x @ self.weight.T
        if "bias" in self:
            y = y + self.bias
        return y
