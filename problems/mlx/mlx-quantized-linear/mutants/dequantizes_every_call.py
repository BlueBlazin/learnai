# BUG: dequantizes the whole matrix on every call and uses an ordinary matmul, so each forward pass allocates the float weight again.
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
        self.weight, self.scales, *biases = mx.quantize(linear.weight, group_size, bits, mode=mode)
        if biases:
            self.biases = biases[0]
        if "bias" in linear:
            self.bias = linear.bias
        self.freeze()

    def __call__(self, x: mx.array) -> mx.array:
        w_hat = mx.dequantize(
            self.weight,
            self.scales,
            self.get("biases"),
            self.group_size,
            self.bits,
            mode=self.mode,
        ).astype(mx.float32)
        y = x @ w_hat.T
        if "bias" in self:
            y = y + self.bias
        return y
