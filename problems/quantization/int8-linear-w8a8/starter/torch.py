import torch

QMIN, QMAX = -128, 127


def int8_linear(
    x: torch.Tensor, w_q: torch.Tensor, w_scale: torch.Tensor, bias: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """A linear layer on int8 weights and dynamically quantized int8 activations.

    x: float32 tensor of shape (n, in); each row is one token and gets its own affine
    int8 grid (scale and zero-point from the row's range, widened to include 0).
    w_q: int8 tensor of shape (out, in), codes in [-127, 127]. w_scale: float32 tensor
    of shape (out,), one scale per row of w_q. bias: float32 tensor of shape (out,).
    Returns (y, acc): y is a float32 tensor of shape (n, out); acc is the int32 tensor
    of shape (n, out) holding the raw integer products x_q @ w_q.T.
    """
    raise NotImplementedError
