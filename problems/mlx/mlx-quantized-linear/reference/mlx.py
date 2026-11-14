import mlx.core as mx

from mlx import nn


class QLinear(nn.Module):
    def __init__(
        self, linear: nn.Linear, group_size: int = 64, bits: int = 4, mode: str = "affine"
    ):
        super().__init__()
        # Plain Python values are attributes of the module, not parameters.
        self.group_size = group_size
        self.bits = bits
        self.mode = mode
        # "affine" returns (codes, scales, biases); "mxfp4" and "mxfp8" return
        # (codes, scales): their scale is a power of two and there is no bias.
        self.weight, self.scales, *biases = mx.quantize(linear.weight, group_size, bits, mode=mode)
        if biases:
            self.biases = biases[0]
        if "bias" in linear:
            self.bias = linear.bias
        # Packed integer codes cannot receive gradients.
        self.freeze()

    def __call__(self, x: mx.array) -> mx.array:
        # x @ W_hat.T straight from the packed codes: the kernel unpacks each code and
        # applies its group's scale and bias as it accumulates. `weight` has shape
        # (out, in * bits / 32), so transpose=True, as for nn.Linear.
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
