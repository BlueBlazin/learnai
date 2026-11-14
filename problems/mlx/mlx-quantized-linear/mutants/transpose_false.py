# BUG: calls mx.quantized_matmul with transpose=False, which computes x @ W_hat and not x @ W_hat.T.
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
        y = mx.quantized_matmul(
            x,
            self.weight,
            scales=self.scales,
            biases=self.get("biases"),
            transpose=False,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
        )
        if "bias" in self:
            y = y + self.bias
        return y
