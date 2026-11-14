import mlx.core as mx

from mlx import nn


class QLinear(nn.Module):
    """A linear layer that stores packed quantization codes and multiplies from them."""

    def __init__(
        self, linear: nn.Linear, group_size: int = 64, bits: int = 4, mode: str = "affine"
    ):
        """Quantize the weight of `linear` with `mx.quantize(w, group_size, bits, mode=mode)`.

        linear: an nn.Linear with a float32 weight of shape (out, in) and, optionally,
            a bias of shape (out,). It is not modified.
        group_size, bits: as for mx.quantize. `in` is a multiple of group_size.
        mode: "affine" (codes, scales, biases) or "mxfp4" / "mxfp8" (codes, scales).
        Parameters afterwards: `weight` (uint32, packed codes), `scales`, `biases`
        (affine only), and `bias` (only if `linear` has one), and nothing else.
        """
        super().__init__()
        raise NotImplementedError

    def __call__(self, x: mx.array) -> mx.array:
        """x: float32 array of shape (..., in). Returns float32 of shape (..., out),
        computed from the packed codes with mx.quantized_matmul, plus the bias."""
        raise NotImplementedError
